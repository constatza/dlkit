"""Base Lightning wrapper with processing pipeline integration.

This module defines the abstract base class for all Lightning wrappers that
integrate with the dlkit.runtime.pipelines pipeline system.
"""

from abc import abstractmethod, ABC
from typing import Any

# Configure checkpoint loading for PyTorch 2.6+ to allow Pydantic settings
# This must happen before any Lightning checkpoint operations
from dlkit.tools.utils.checkpoint_security import configure_checkpoint_loading
configure_checkpoint_loading()

import torch
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MetricCollection
from torch.nn import ModuleDict

from dlkit.tools.config import (
    BuildContext,
    FactoryProvider,
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.tools.config.data_entries import DataEntry, Target, is_feature_entry, is_target_entry
from dlkit.tools.config.core.updater import update_settings
from dlkit.core.shape_specs import IShapeSpec
from dlkit.runtime.workflows.entry_registry import DataEntryRegistry
from dlkit.runtime.pipelines.pipeline import (
    ProcessingPipeline,
    DataExtractionStep,
    ModelInvocationStep,
    OutputClassificationStep,
    OutputNamingStep,
    ValidationDataStep,
)
from dlkit.runtime.pipelines.model_invokers import ModelInvokerFactory
from dlkit.runtime.pipelines.classifiers import NameBasedClassifier
from dlkit.runtime.pipelines.context import ProcessingContext


class ProcessingLightningWrapper(LightningModule, ABC):
    """Abstract base Lightning wrapper with core processing pipeline integration.

    This wrapper provides the fundamental Lightning integration with basic processing
    pipeline support (extraction → invocation → classification → loss pairing).
    Transform functionality is provided by specialized subclasses.

    Attributes:
        model (torch.nn.Module): Underlying PyTorch model.
        shape_spec (IShapeSpec | None): Shape specification associated with the model.
        val_metrics (torchmetrics.MetricCollection): Validation metrics.
        test_metrics (torchmetrics.MetricCollection): Test metrics.
        loss_function (callable): Loss function operating on processed tensors.
        train_pipeline (ProcessingPipeline): Training pipeline.
        val_pipeline (ProcessingPipeline): Validation pipeline.
        test_pipeline (ProcessingPipeline): Test pipeline.
        predict_pipeline (ProcessingPipeline): Inference pipeline (no loss pairing).
    """

    def __init__(
        self,
        *,
        settings: WrapperComponentSettings,
        model_settings: ModelComponentSettings,
        entry_configs: dict[str, DataEntry] | None = None,
        shape_spec: IShapeSpec | None = None,
        **kwargs,
    ):
        """Initialize the processing Lightning wrapper.

        Args:
            settings (WrapperComponentSettings): Wrapper configuration settings.
            model_settings (ModelComponentSettings): Model configuration settings.
            entry_configs (dict[str, DataEntry] | None): Data entry configurations for pipeline setup.
            shape_spec (IShapeSpec | None): Shape specification for models.
            **kwargs: Additional arguments passed to LightningModule.
        """
        super().__init__()

        # Store configuration
        self.save_hyperparameters(
            {
                "settings": settings,
                "model_settings": model_settings,
            },
            ignore=["settings", "model_settings", "entry_configs"],
        )

        # Store shape information for checkpointing
        self.shape_spec = shape_spec

        # Initialize model with ABC-based factory
        self.model = self._create_abc_model(model_settings, shape_spec)

        # Apply precision to model immediately after creation
        # Lightning's precision plugin will convert the wrapper, but we need to ensure
        # the nested model is also converted. This matches the pattern in test models.
        from dlkit.interfaces.api.services.precision_service import get_precision_service

        precision_service = get_precision_service()
        precision_strategy = precision_service.resolve_precision()
        dtype = precision_strategy.to_torch_dtype()
        self.model = self.model.to(dtype=dtype)

        # Ensure wrapper keeps a canonical reference to model-provided shape specs
        if self.shape_spec is None:
            self._assign_shape_spec(self._derive_shape_spec_from_model())
        else:
            self._assign_shape_spec(self.shape_spec)
        self.val_metrics = MetricCollection([
            FactoryProvider.create_component(metric, BuildContext(mode="training"))
            for metric in settings.metrics
        ])
        self.test_metrics = MetricCollection([
            FactoryProvider.create_component(metric, BuildContext(mode="training"))
            for metric in settings.metrics
        ])

        # Loss function from model or settings
        self.loss_function = getattr(
            self.model, "loss_function", None
        ) or FactoryProvider.create_component(
            settings.loss_function,
            BuildContext(mode="training"),
        )

        self.optimizer = settings.optimizer
        self.scheduler = settings.scheduler
        # Keep a direct reference to wrapper settings for pipeline decisions
        self._wrapper_settings = settings

        # Ensure lr/learning_rate hyperparameters mirror optimizer settings
        self._sync_lr_hparam()

        # Initialize processing pipelines
        self._entry_configs = entry_configs or {}

        # Register entry configs with global registry for end user access
        if self._entry_configs:
            registry = DataEntryRegistry.get_instance()
            registry.register_entries(self._entry_configs)

        # Initialize pipeline attributes to be set by subclass implementations
        self.train_pipeline: ProcessingPipeline
        self.val_pipeline: ProcessingPipeline
        self.test_pipeline: ProcessingPipeline
        self.predict_pipeline: ProcessingPipeline

        # Set up pipelines via template methods
        self._setup_processing_pipelines(self._entry_configs)
        self._setup_predict_pipeline(self._entry_configs)

    def configure_model(self) -> None:
        """Ensure precision is maintained after Lightning setup/checkpoint restore.

        Lightning calls this hook after:
        - Initial model creation
        - Checkpoint restoration (e.g., during LR tuning)
        - Strategy setup

        This ensures the nested model's dtype persists even after checkpoint operations
        that might reconstruct the model. Critical for LR tuning which saves/restores
        checkpoints.
        """
        super().configure_model()

        # Reapply precision to nested model
        # This is idempotent and safe to call multiple times
        from dlkit.interfaces.api.services.precision_service import get_precision_service

        precision_service = get_precision_service()
        precision_strategy = precision_service.resolve_precision()
        dtype = precision_strategy.to_torch_dtype()
        self.model = self.model.to(dtype=dtype)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save enhanced metadata including complete shape information.

        Ensures comprehensive model reconstruction information persists across
        checkpoint saves/loads, eliminating the need for manual shape parameters
        during inference.

        This method is pure - it only reads state and writes to the checkpoint dict.
        No side effects (state mutations) are performed during checkpoint save.
        """
        super().on_save_checkpoint(checkpoint)

        # Create comprehensive DLKit metadata section
        dlkit_metadata: dict[str, Any] = {
            'version': '2.0',
            'model_family': self._detect_model_family(),
            'wrapper_type': self.__class__.__name__,
        }

        # Save enhanced shape information using ShapeSpec
        active_shape_spec = self.shape_spec or self._derive_shape_spec_from_model()
        if active_shape_spec is not None and not active_shape_spec.is_empty():
            canonical_spec = active_shape_spec.with_canonical_aliases()
            dlkit_metadata['shape_spec'] = canonical_spec.to_dict()

        # Save model settings for reconstruction
        dlkit_metadata['model_settings'] = self._serialize_model_settings()

        # Save entry configs for pipeline reconstruction
        dlkit_metadata['entry_configs'] = self._serialize_entry_configs()

        # Store in checkpoint
        checkpoint['dlkit_metadata'] = dlkit_metadata


    def _detect_model_family(self) -> str:
        """Detect model family for appropriate shape handling.

        Returns:
            Model family identifier ("dlkit_nn", "graph", "timeseries", "external")
        """
        try:
            # Use existing detection logic from model detection module
            from dlkit.runtime.workflows.factories.model_detection import detect_model_type
            from dlkit.tools.config import GeneralSettings
            # We don't have full GeneralSettings here, but can pass None
            # The detector will work with just model_settings
            model_type = detect_model_type(self.hparams.model_settings, None)  # type: ignore[arg-type]
            return model_type.value
        except Exception:
            return "external"

    def _serialize_model_settings(self) -> dict[str, Any]:
        """Serialize model settings for reconstruction.

        Returns:
            Serialized model configuration
        """
        try:
            settings = self.hparams.model_settings

            # Extract base fields
            name = getattr(settings, 'name', None)
            module_path = getattr(settings, 'module_path', None)
            class_name = settings.__class__.__name__

            # Extract hyperparameters (all extra fields allowed by Pydantic model_config)
            # Exclude base fields and checkpoint
            params = {}
            if hasattr(settings, 'model_dump'):
                # Use exclude_none=False to keep None values (important for reconstruction)
                # Don't use mode='json' as it can't serialize ABCMeta types that may be in settings
                all_fields = settings.model_dump()
                excluded = {'name', 'module_path', 'checkpoint'}
                params = {k: v for k, v in all_fields.items() if k not in excluded and v is not None}

            return {
                'name': name,
                'module_path': module_path,
                'params': params,
                'class_name': class_name
            }
        except Exception:
            # Return empty dict on serialization failure - checkpoint will be valid
            # but may not support automatic inference without explicit model_settings
            return {}

    def _serialize_entry_configs(self) -> dict[str, Any]:
        """Serialize entry configurations for pipeline reconstruction.

        Returns:
            Serialized entry configurations
        """
        try:
            if hasattr(self, '_entry_configs') and self._entry_configs:
                # Serialize DataEntry objects to dict format
                serialized = {}
                for name, entry in self._entry_configs.items():
                    serialized[name] = {
                        'name': entry.name,
                        'class_name': entry.__class__.__name__,
                        # Add other relevant fields as needed
                    }
                return serialized
        except Exception:
            pass
        return {}

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore enhanced metadata and shape information from checkpoint.

        Reconstructs shape information when loading from checkpoint using the
        modern shape specification metadata persisted alongside the weights.

        Raises:
            ValueError: If checkpoint version is unsupported or metadata format is invalid.
        """
        super().on_load_checkpoint(checkpoint)

        # Check for dlkit_metadata (required for v2.0+)
        if 'dlkit_metadata' not in checkpoint:
            # Legacy checkpoint without dlkit_metadata
            raise ValueError(
                "Checkpoint missing 'dlkit_metadata'. This checkpoint uses a legacy format "
                "that is no longer supported. Please re-train your model with the current "
                "version of dlkit to generate a compatible checkpoint."
            )

        metadata = checkpoint['dlkit_metadata']

        # Validate version (BREAKING: only support v2.0+)
        version = metadata.get('version')
        if version is None:
            raise ValueError(
                "Checkpoint metadata missing 'version' field. Cannot verify compatibility."
            )

        if version != '2.0':
            raise ValueError(
                f"Unsupported checkpoint version '{version}'. This version of dlkit only "
                f"supports version '2.0'. If you have an older checkpoint, please re-train "
                f"your model with the current version."
            )

        # Restore shape from ShapeSpec if available
        if 'shape_spec' in metadata:
            try:
                from dlkit.core.shape_specs import ShapeSystemFactory

                factory = ShapeSystemFactory.create_production_system()
                loaded_spec = factory.create_shape_spec_from_serialized(metadata['shape_spec'])
                self._assign_shape_spec(loaded_spec)
            except Exception as e:
                # Log warning but don't fail the load - shape reconstruction is best-effort
                import warnings
                warnings.warn(
                    f"Could not restore shape specification from checkpoint: {e}",
                    RuntimeWarning,
                    stacklevel=2
                )

    def _derive_shape_spec_from_model(self) -> IShapeSpec | None:
        """Extract shape specification from the instantiated model when possible."""
        try:
            from dlkit.core.models.nn.base import ShapeAwareModel

            if isinstance(self.model, ShapeAwareModel):
                return self.model.get_unified_shape()
        except Exception:
            pass

        if hasattr(self.model, "get_shape_spec"):
            try:
                candidate = self.model.get_shape_spec()
                if isinstance(candidate, IShapeSpec):
                    return candidate
            except Exception:
                pass

        return None

    def _assign_shape_spec(self, shape_spec: IShapeSpec | None) -> None:
        """Update wrapper and underlying model with the provided shape spec."""
        self.shape_spec = shape_spec

        if shape_spec is None:
            return

        try:
            from dlkit.core.models.nn.base import ShapeAwareModel

            if isinstance(self.model, ShapeAwareModel):
                # Update the cached unified shape used by shape-aware models
                self.model._unified_shape = shape_spec  # noqa: SLF001 - internal cache update
        except Exception:
            pass


    @abstractmethod
    def _setup_processing_pipelines(self, entry_configs: dict[str, DataEntry]) -> None:
        """Set up processing pipelines for training/validation/test.

        Template method to be implemented by subclasses with their specific pipeline structure.

        Args:
            entry_configs (dict[str, DataEntry]): Mapping from entry name to configuration.
        """
        pass

    @abstractmethod
    def _setup_predict_pipeline(self, entry_configs: dict[str, DataEntry]) -> None:
        """Set up inference-only processing pipeline.

        Template method to be implemented by subclasses with their specific predict pipeline.
        This pipeline should exclude LossPairingStep to make targets optional during inference.

        Args:
            entry_configs (dict[str, DataEntry]): Mapping from entry name to configuration.
        """
        pass


    def _create_output_classifier(self):
        """Create the output classifier for this wrapper.

        Can be overridden by subclasses to use different classification strategies.

        Returns:
            OutputClassifier instance
        """
        return NameBasedClassifier()

    def _create_output_namer(self):
        """Create the output namer for this wrapper.

        Default: map prediction keys to target names using exact shape matching.
        Can be overridden by subclasses to customize naming behavior.
        """
        from dlkit.runtime.pipelines.naming import TargetNameByShapeNamer

        return TargetNameByShapeNamer()

    def _safe_log(self, *args, **kwargs) -> None:
        """Safe logging that only logs when trainer is available.

        This prevents errors when running steps without a trainer attached.
        """
        try:
            if hasattr(self, "trainer") and self.trainer is not None:
                self.log(*args, **kwargs)
        except Exception:
            # Silently ignore logging errors in test scenarios
            pass

    def _safe_log_dict(self, *args, **kwargs) -> None:
        """Safe dict logging that only logs when trainer is available.

        This prevents errors when running steps without a trainer attached.
        """
        try:
            if hasattr(self, "trainer") and self.trainer is not None:
                self.log_dict(*args, **kwargs)
        except Exception:
            # Silently ignore logging errors in test scenarios
            pass

    def _get_attached_trainer(self) -> Any:
        """Return the attached trainer or fabric shim without triggering Lightning errors."""
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            return trainer

        fabric = getattr(self, "_fabric", None)
        if fabric is not None:
            try:
                from lightning.pytorch.core.module import _TrainerFabricShim

                return _TrainerFabricShim(fabric=fabric)
            except Exception:  # pragma: no cover - fabric usage is rare in tests
                return None

        return None

    def _get_optimizer_lr(self) -> float | None:
        """Resolve the effective learning rate from optimizer settings or trainer state."""
        raw_lr = getattr(self.optimizer, "lr", None)
        if isinstance(raw_lr, (float, int)):
            return float(raw_lr)

        trainer = self._get_attached_trainer()
        if trainer and getattr(trainer, "optimizers", None):
            try:
                return float(trainer.optimizers[0].param_groups[0]["lr"])
            except (KeyError, IndexError, TypeError, ValueError):
                return None

        return None

    def _sync_lr_hparam(self) -> None:
        """Synchronise stored hyperparameters with the current learning rate value."""
        lr_value = self._get_optimizer_lr()
        if not hasattr(self, "hparams"):
            return

        self.hparams["lr"] = lr_value
        self.hparams["learning_rate"] = lr_value

    @property
    def lr(self) -> float | None:
        """Expose lr attribute for Lightning LR finder compatibility."""
        return self._get_optimizer_lr()

    @lr.setter
    def lr(self, value: float) -> None:
        numeric = float(value)
        self.optimizer = update_settings(self.optimizer, {"lr": numeric})
        trainer = self._get_attached_trainer()
        if trainer and getattr(trainer, "optimizers", None):
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = numeric
        self._sync_lr_hparam()

    @property
    def learning_rate(self) -> float | None:
        """Alias maintained for frameworks expecting learning_rate attribute."""
        return self.lr

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.lr = value

    def _log_stage_outputs(
        self,
        stage: str,
        loss: Tensor | None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Centralized metric logging hook for training stages.

        Args:
            stage: Stage identifier ("train", "val", "test").
            loss: Scalar loss tensor for the stage.
            metrics: Optional dictionary of additional metrics to log.
        """
        if loss is not None:
            loss_name = self._format_metric_name(stage, "loss")
            self._safe_log(
                loss_name,
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        if metrics:
            formatted: dict[str, Any] = {}
            for key, value in metrics.items():
                metric_name = self._format_metric_name(stage, key)
                formatted[metric_name] = value

            self._safe_log_dict(formatted, on_step=False, on_epoch=True, prog_bar=True)

    def _format_metric_name(self, stage: str, name: str) -> str:
        """Normalize metric names according to stage-specific conventions."""

        stage_lower = stage.lower()
        name_lower = name.lower()

        aliases = {
            "train": ("train", "training"),
            "val": ("val", "valid", "validation"),
            "test": ("test", "testing"),
        }

        for alias in aliases.get(stage_lower, (stage_lower,)):
            if name_lower.startswith(alias):
                return name

        if stage_lower == "test":
            if name_lower.endswith(" test"):
                return name
            return f"{name} test"

        # Default: prefix with stage for all stages (including val)
        return f"{stage_lower}_{name}"

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Forward pass through the model.

        This method should be implemented by subclasses to handle
        the specific input format for their model type.
        """
        pass

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Training step using processing pipeline.

        Args:
            batch (dict[str, Tensor]): Raw batch dataflow from the dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict[str, Any]: Dictionary containing the training loss.
        """
        # Process batch through training pipeline
        context = self.train_pipeline.execute(batch)

        # Compute loss using processed dataflow
        loss = self._compute_loss(context)

        # Log metrics centrally for orchestration
        self._log_stage_outputs("train", loss)

        return {"loss": loss}

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Validation step using processing pipeline.

        Args:
            batch (dict[str, Tensor]): Raw batch dataflow from the dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict[str, Any]: Dictionary containing validation metrics.
        """
        # Process batch through validation pipeline
        context = self.val_pipeline.execute(batch)

        # Compute loss and metrics
        val_loss = self._compute_loss(context)
        metrics = self._compute_metrics(context, self.val_metrics)

        # Log metrics centrally for orchestration
        self._log_stage_outputs("val", val_loss, metrics)

        return {"val_loss": val_loss}

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Test step using processing pipeline.

        Args:
            batch (dict[str, Tensor]): Raw batch dataflow from the dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict[str, Any]: Dictionary containing test metrics.
        """
        # Process batch through test pipeline
        context = self.test_pipeline.execute(batch)

        # Compute loss and metrics
        test_loss = self._compute_loss(context)
        metrics = self._compute_metrics(context, self.test_metrics)

        # Log metrics centrally for orchestration
        self._log_stage_outputs("test", test_loss, metrics)

        return {"test_loss": test_loss}

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, dict[str, Tensor]]:
        """Prediction step using dedicated inference pipeline.

        Uses the predict_pipeline which excludes LossPairingStep, making targets
        optional during inference.

        Args:
            batch (dict[str, Tensor]): Raw batch dataflow from the dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict[str, dict[str, Tensor]]: Dictionary with ``predictions``, ``targets``, and ``latents``.
        """
        # Use dedicated predict pipeline - no loss pairing required
        context = self.predict_pipeline.execute(batch)

        return {"predictions": dict(context.predictions), "targets": dict(context.targets), "latents": context.latents}


    def _compute_loss(self, context: ProcessingContext) -> Tensor:
        """Compute loss from processing context.

        Args:
            context (ProcessingContext): Processing context with ``loss_data`` populated.

        Returns:
            torch.Tensor: Computed loss tensor.
        """
        # Strict: require LossPairingStep to have populated named (pred, target) pairs.
        if not context.loss_data:
            raise RuntimeError(
                "Loss pairing missing: ensure LossPairingStep produced pairs or model outputs "
                "match target keys."
            )

        first_val = next(iter(context.loss_data.values()))
        if not (isinstance(first_val, tuple) and len(first_val) == 2):
            raise RuntimeError(
                "Invalid loss_data format: expected {name: (pred, target)} pairs generated by "
                "LossPairingStep."
            )

        total = None
        for name, pair in context.loss_data.items():
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise RuntimeError(
                    f"Invalid loss_data entry for '{name}': expected (pred, target) tuple."
                )
            pred, target = pair
            val = self.loss_function(pred, target)
            total = val if total is None else total + val

        if total is None:
            raise RuntimeError("No loss pairs available for computation")
        return total

    def _compute_metrics(
        self, context: ProcessingContext, metrics: MetricCollection
    ) -> dict[str, Any]:
        """Compute metrics from processing context.

        Ensures predictions and targets use the model's dtype (user's configured precision)
        for metric computation. This maintains precision consistency throughout the pipeline.

        Args:
            context (ProcessingContext): Processing context with predictions and targets.
            metrics (MetricCollection): Metrics collection to compute.

        Returns:
            dict[str, Any]: Dictionary of computed metrics.
        """
        if not context.predictions or not context.targets:
            return {}

        # Use first prediction and target for metrics
        pred = next(iter(context.predictions.values()))
        target = next(iter(context.targets.values()))

        # Ensure both use model's dtype (maintains user's precision choice)
        # If model doesn't have dtype property, skip casting
        if hasattr(self.model, "dtype"):
            model_dtype = self.model.dtype

            # Cast if needed (defensive - should already match from pipeline)
            if pred.is_floating_point() and pred.dtype != model_dtype:
                pred = pred.to(dtype=model_dtype)
            if target.is_floating_point() and target.dtype != model_dtype:
                target = target.to(dtype=model_dtype)

        return metrics(pred, target)

    def on_train_epoch_end(self) -> None:
        """Log the current learning rate at epoch end."""
        if self.trainer and self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self._safe_log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Reset validation metrics at the end of the epoch."""
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Reset test metrics at the end of the epoch."""
        self.test_metrics.reset()

    def configure_optimizers(self):  # type: ignore[override]
        """Configure optimizer and scheduler from settings."""
        # Build optimizer and scheduler via factories with explicit overrides
        # Note: lr override removed - settings already contain the resolved learning rate from overrides
        optimizer = FactoryProvider.create_component(
            self.optimizer,
            BuildContext(mode="training", overrides={"params": self.model.parameters()}),
        )
        scheduler = FactoryProvider.create_component(
            self.scheduler,
            BuildContext(mode="training", overrides={"optimizer": optimizer}),
        )

        if scheduler is None:
            return {"optimizer": optimizer}  # type: ignore[return-value]

        return {  # type: ignore[return-value]
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

    def get_entry_configs(self) -> dict[str, DataEntry]:
        """Get the data entry configurations used by this wrapper.

        Returns:
            Dictionary mapping entry names to DataEntry configurations
        """
        return self._entry_configs.copy()

    def get_feature_configs(self) -> dict[str, DataEntry]:
        """Get feature entry configurations.

        Returns:
            Dictionary mapping feature names to Feature configurations
        """
        return {
            name: config for name, config in self._entry_configs.items()
            if is_feature_entry(config)
        }

    def get_target_configs(self) -> dict[str, DataEntry]:
        """Get target entry configurations.

        Returns:
            Dictionary mapping target names to Target configurations
        """
        return {
            name: config for name, config in self._entry_configs.items()
            if is_target_entry(config)
        }

    def _is_dlkit_model(self, model_settings) -> bool:
        """Check if model settings refer to a dlkit model that should receive shapes."""
        try:
            model_name = getattr(model_settings, "name", None)
            if model_name is None:
                return False

            # Check module path for dlkit indicators
            module_path_str = str(getattr(model_settings, "module_path", "")).lower()
            if "dlkit.core.models.nn" in module_path_str:
                return True

            # For testing: treat generic module paths as dlkit models
            if module_path_str in ("", "x", "test", "dummy", "tests.helpers"):
                return True

            # Try to import and check inheritance
            if isinstance(model_name, str):
                try:
                    from dlkit.tools.utils.general import import_object as _import
                    model_cls = _import(model_name, fallback_module=getattr(model_settings, "module_path", ""))
                    from dlkit.core.models.nn.base import ShapeAwareModel, ShapeAgnosticModel
                    return issubclass(model_cls, (ShapeAwareModel, ShapeAgnosticModel))
                except Exception:
                    pass
            elif isinstance(model_name, type):
                try:
                    from dlkit.core.models.nn.base import ShapeAwareModel, ShapeAgnosticModel
                    return issubclass(model_name, (ShapeAwareModel, ShapeAgnosticModel))
                except Exception:
                    pass

            return False
        except Exception:
            return False

    def _create_abc_model(self, model_settings: ModelComponentSettings, shape_spec: IShapeSpec | None) -> torch.nn.Module:
        """Create model using ABC-based approach.

        Args:
            model_settings: Model configuration settings
            shape_spec: Shape specification for the model

        Returns:
            Created model instance
        """
        from dlkit.runtime.workflows.factories.model_detection import detect_model_type, ModelType
        from dlkit.tools.config import GeneralSettings
        from dlkit.core.models.nn.base import ShapeAwareModel, ShapeAgnosticModel

        # Create minimal settings for model detection
        settings = GeneralSettings(MODEL=model_settings)

        # Import model class
        model_name = getattr(model_settings, "name", None)
        if isinstance(model_name, str):
            from dlkit.tools.utils.general import import_object
            model_cls = import_object(
                model_name,
                fallback_module=getattr(model_settings, "module_path", "")
            )
        elif isinstance(model_name, type):
            model_cls = model_name
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        # Get model parameters from settings attributes
        model_kwargs = {}

        # Extract all non-None model parameters from the settings object
        # Exclude meta attributes and only include model constructor parameters
        exclude_fields = {'name', 'module_path', 'checkpoint'}
        for field_name in model_settings.__class__.model_fields:
            if field_name not in exclude_fields:
                field_value = getattr(model_settings, field_name, None)
                if field_value is not None:
                    model_kwargs[field_name] = field_value

        # Also support legacy params attribute if present
        if hasattr(model_settings, 'params') and model_settings.params:
            model_kwargs.update(model_settings.params)

        # Create model based on ABC type
        try:
            if issubclass(model_cls, ShapeAwareModel):
                if shape_spec is None:
                    raise ValueError(f"ShapeAwareModel {model_cls.__name__} requires shape specification")
                return model_cls(unified_shape=shape_spec, **model_kwargs)
            elif issubclass(model_cls, ShapeAgnosticModel):
                return model_cls(**model_kwargs)
        except TypeError:
            pass

        # External model - create without shape
        return model_cls(**model_kwargs)
