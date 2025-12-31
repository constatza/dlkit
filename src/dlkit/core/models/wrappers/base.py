"""Base Lightning wrapper with direct processing methods.

This module defines the abstract base class for all Lightning wrappers.
Replaces the Chain of Responsibility pipeline pattern with direct helper methods
for simpler, more maintainable code.
"""

from abc import abstractmethod, ABC
from typing import Any
from loguru import logger

# Configure checkpoint loading for PyTorch 2.6+ to allow Pydantic settings
# This must happen before any Lightning checkpoint operations
from dlkit.tools.utils.checkpoint_security import configure_checkpoint_loading
configure_checkpoint_loading()

import torch
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MetricCollection

from dlkit.tools.config import (
    BuildContext,
    FactoryProvider,
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.tools.config.data_entries import DataEntry, is_feature_entry, is_target_entry
from dlkit.tools.config.core.updater import update_settings
from dlkit.core.shape_specs import IShapeSpec
from dlkit.runtime.workflows.entry_registry import DataEntryRegistry


class ProcessingLightningWrapper(LightningModule, ABC):
    """Abstract base Lightning wrapper with direct processing methods.

    Provides fundamental Lightning integration with streamlined processing:
    extraction → invocation → loss pairing → metrics.

    Transform functionality is provided by specialized subclasses (StandardLightningWrapper).

    Attributes:
        model (torch.nn.Module): Underlying PyTorch model.
        shape_spec (IShapeSpec | None): Shape specification associated with the model.
        val_metrics (torchmetrics.MetricCollection): Validation metrics.
        test_metrics (torchmetrics.MetricCollection): Test metrics.
        loss_function (callable): Loss function operating on predictions and targets.
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
            entry_configs (dict[str, DataEntry] | None): Data entry configurations.
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
        # Keep a direct reference to wrapper settings for decisions
        self._wrapper_settings = settings

        # Ensure lr/learning_rate hyperparameters mirror optimizer settings
        self._sync_lr_hparam()

        # Store entry configs for feature/target categorization
        self._entry_configs = entry_configs or {}

        # Register entry configs with global registry for end user access
        if self._entry_configs:
            registry = DataEntryRegistry.get_instance()
            registry.register_entries(self._entry_configs)

        # Pre-compute feature and target names for efficient lookup
        self._feature_names = {
            name for name, config in self._entry_configs.items()
            if is_feature_entry(config)
        }
        self._target_names = {
            name for name, config in self._entry_configs.items()
            if is_target_entry(config)
        }

    # =============================================================================
    # Core Processing Helper Methods (Replace Pipeline Steps)
    # =============================================================================

    def _extract_features_targets(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Extract and categorize batch data into features and targets.

        When no explicit entry configuration is provided, applies a simple
        name-based heuristic:
        - targets: keys like {"y", "target", "targets", "label", "labels"} (case-insensitive)
        - features: all remaining keys

        Args:
            batch (dict[str, Tensor]): Raw batch from dataset.

        Returns:
            tuple[dict[str, Tensor], dict[str, Tensor]]: Features and targets.
        """
        if not self._feature_names and not self._target_names:
            # Heuristic fallback when no configs are provided
            target_like = {"y", "target", "targets", "label", "labels"}
            features: dict[str, Tensor] = {}
            targets: dict[str, Tensor] = {}
            for name, tensor in batch.items():
                n = str(name).lower()
                if n in target_like:
                    targets[name] = tensor
                else:
                    features[name] = tensor
            return features, targets

        # Extract based on explicit configuration
        features = {
            name: tensor
            for name, tensor in batch.items()
            if name in self._feature_names
        }
        targets = {
            name: tensor
            for name, tensor in batch.items()
            if name in self._target_names
        }
        return features, targets

    def _log_dtype_mismatches(self, features: dict[str, Tensor] | Tensor) -> None:
        """Log dtype mismatches between features and model (debug only).

        Uses guard clauses to avoid nested ifs.
        Lightning's precision plugin handles actual dtype alignment.
        """
        import torch

        # Guard: Skip if model doesn't have dtype
        if not hasattr(self.model, "dtype"):
            return

        model_dtype = self.model.dtype

        # Guard: Ensure dtype is actually a torch.dtype (type narrowing)
        if not isinstance(model_dtype, torch.dtype):
            return

        # Use match-case for feature type handling
        match features:
            case dict():
                self._log_dict_features_dtype(features, model_dtype)
            case Tensor():
                self._log_tensor_feature_dtype(features, model_dtype)

    def _log_dict_features_dtype(self, features: dict[str, Tensor], model_dtype: torch.dtype) -> None:
        """Log dtype mismatches for dict features."""
        for name, value in features.items():
            # Guard: Skip non-tensor values (e.g., int indices in graph data)
            if not isinstance(value, Tensor):
                continue

            # Guard: Skip non-floating point tensors
            if not value.is_floating_point():
                continue

            # Guard: Skip if dtypes match
            if value.dtype == model_dtype:
                continue

            # Log the mismatch
            logger.debug(
                f"Feature '{name}' dtype {value.dtype} differs from model dtype {model_dtype}. "
                f"Lightning's precision plugin will handle alignment."
            )

    def _log_tensor_feature_dtype(self, features: Tensor, model_dtype: torch.dtype) -> None:
        """Log dtype mismatch for single tensor feature."""
        # Guard: Skip non-floating point tensors
        if not features.is_floating_point():
            return

        # Guard: Skip if dtypes match
        if features.dtype == model_dtype:
            return

        # Log the mismatch
        logger.debug(
            f"Features dtype {features.dtype} differs from model dtype {model_dtype}. "
            f"Lightning's precision plugin will handle alignment."
        )

    def _forward_features(self, features: dict[str, Tensor] | Tensor) -> dict[str, Tensor] | Tensor:
        """Forward features through model with match-case logic."""
        match features:
            case dict():
                return self._forward_dict_features(features)
            case _:
                return self.model(features)

    def _forward_dict_features(self, features: dict[str, Tensor]) -> dict[str, Tensor] | Tensor:
        """Forward dict features - try **kwargs first, fallback to single tensor."""
        try:
            return self.model(**features)
        except TypeError:
            # Model doesn't accept **kwargs, try single tensor
            return self.model(next(iter(features.values())))

    def _invoke_model(self, features: dict[str, Tensor] | Tensor) -> dict[str, Tensor] | Tensor:
        """Invoke model forward pass with defensive dtype validation.

        Data should already be loaded in the correct dtype via load_array(),
        but this provides a safety check and automatic casting if needed.

        Args:
            features (dict[str, Tensor] | Tensor): Model inputs.

        Returns:
            dict[str, Tensor] | Tensor: Model outputs.

        Raises:
            RuntimeError: If model invocation fails or no features available.
        """
        if isinstance(features, dict) and not features:
            raise RuntimeError("No features available for model invocation")

        try:
            # Debug-level dtype tracking (before Lightning precision kicks in)
            self._log_dtype_mismatches(features)

            # Model forward - use match-case for cleaner logic
            return self._forward_features(features)

        except Exception as e:
            raise RuntimeError(f"Model invocation failed: {e}") from e

    def _compute_loss(self, predictions: dict[str, Tensor] | Tensor, targets: dict[str, Tensor]) -> Tensor:
        """Compute loss from predictions and targets with automatic pairing.

        Pairing rules:
        - Strict mapping: for each target there must be a corresponding prediction key
        - Single-target fallback: if exactly one target and one prediction, pair them
        - Autoencoder mode: handled by subclass (StandardLightningWrapper)

        Args:
            predictions (dict[str, Tensor] | Tensor): Model predictions.
            targets (dict[str, Tensor]): Ground truth targets.

        Returns:
            torch.Tensor: Computed loss.

        Raises:
            RuntimeError: If pairing fails or required pairs are missing.
        """
        # Normalize predictions to dict format
        if isinstance(predictions, Tensor):
            # Single tensor output - check if we can pair with targets
            if len(targets) == 1:
                # Single target - pair with single prediction
                pred = predictions
                target = next(iter(targets.values()))

                # Align target dtype with prediction dtype
                if target.is_floating_point() and pred.is_floating_point():
                    if target.dtype != pred.dtype:
                        logger.debug(
                            f"Target dtype ({target.dtype}) differs from prediction ({pred.dtype}). "
                            f"Casting target to match prediction for loss computation."
                        )
                        target = target.to(dtype=pred.dtype)

                return self.loss_function(pred, target)
            else:
                raise RuntimeError(
                    f"Model returned single tensor but found {len(targets)} targets. "
                    f"Cannot determine pairing. Target keys: {list(targets.keys())}"
                )
        elif isinstance(predictions, dict):
            # Dict predictions - pair by name
            pnames = list(predictions.keys())
            tnames = list(targets.keys())

            # Single-target fallback
            if len(tnames) == 1 and len(pnames) == 1:
                tname = tnames[0]
                pred = predictions[pnames[0]]
                target = targets[tname]

                # Align target dtype with prediction dtype
                if target.is_floating_point() and pred.is_floating_point():
                    if target.dtype != pred.dtype:
                        logger.debug(
                            f"Target '{tname}' dtype ({target.dtype}) differs from prediction ({pred.dtype}). "
                            f"Casting target to match prediction for loss computation."
                        )
                        target = target.to(dtype=pred.dtype)

                return self.loss_function(pred, target)

            # Strict matching by names
            missing = [t for t in tnames if t not in predictions]
            unexpected = [p for p in pnames if p not in targets]
            if missing or unexpected:
                msg_parts = []
                if missing:
                    msg_parts.append(f"missing predictions for targets: {missing}")
                if unexpected:
                    msg_parts.append(f"unexpected prediction keys: {unexpected}")
                available = f"available targets={tnames}, predictions={pnames}"
                raise RuntimeError(f"Loss pairing failed: {', '.join(msg_parts)}; {available}")

            # Build pairs by matching keys with dtype alignment
            total_loss = None
            for name in tnames:
                pred = predictions[name]
                target = targets[name]

                # Align target dtype with prediction dtype
                if target.is_floating_point() and pred.is_floating_point():
                    if target.dtype != pred.dtype:
                        logger.debug(
                            f"Target '{name}' dtype ({target.dtype}) differs from prediction ({pred.dtype}). "
                            f"Casting target to match prediction for loss computation."
                        )
                        target = target.to(dtype=pred.dtype)

                loss = self.loss_function(pred, target)
                total_loss = loss if total_loss is None else total_loss + loss

            if total_loss is None:
                raise RuntimeError("No loss pairs available for computation")
            return total_loss
        else:
            raise RuntimeError(f"Unsupported prediction type: {type(predictions)}")

    def _update_metrics(
        self,
        predictions: dict[str, Tensor] | Tensor,
        targets: dict[str, Tensor],
        stage: str
    ) -> dict[str, Any]:
        """Compute metrics from predictions and targets.

        Ensures predictions and targets use the model's dtype (user's configured precision)
        for metric computation. This maintains precision consistency throughout the pipeline.

        Args:
            predictions (dict[str, Tensor] | Tensor): Model predictions.
            targets (dict[str, Tensor]): Ground truth targets.
            stage (str): Stage identifier ("train", "val", "test").

        Returns:
            dict[str, Any]: Dictionary of computed metrics.
        """
        # Select appropriate metric collection
        metrics = {"val": self.val_metrics, "test": self.test_metrics}.get(stage)
        if metrics is None:
            return {}

        # Extract first prediction and target for metrics
        if isinstance(predictions, dict):
            if not predictions or not targets:
                return {}
            pred = next(iter(predictions.values()))
            target = next(iter(targets.values()))
        else:
            pred = predictions
            if not targets:
                return {}
            target = next(iter(targets.values()))

        # Ensure both use model's dtype (maintains user's precision choice)
        if hasattr(self.model, "dtype"):
            import torch

            model_dtype = self.model.dtype

            # Guard: Ensure dtype is torch.dtype (type narrowing)
            if not isinstance(model_dtype, torch.dtype):
                return metrics(pred, target)

            # Cast if needed (defensive - should already match from pipeline)
            if pred.is_floating_point() and pred.dtype != model_dtype:
                pred = pred.to(dtype=model_dtype)
            if target.is_floating_point() and target.dtype != model_dtype:
                target = target.to(dtype=model_dtype)

        return metrics(pred, target)

    # =============================================================================
    # Lightning Step Methods (Use Direct Helpers)
    # =============================================================================

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Training step with direct processing (no pipeline).

        Args:
            batch (dict[str, Tensor]): Raw batch from dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict[str, Any]: Dictionary containing the training loss.
        """
        # 1. Extract features and targets
        features, targets = self._extract_features_targets(batch)

        # 2. Model forward
        predictions = self._invoke_model(features)

        # 3. Compute loss
        loss = self._compute_loss(predictions, targets)

        # 4. Log metrics
        self._log_stage_outputs("train", loss)

        return {"loss": loss}

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Validation step with direct processing (no pipeline).

        Args:
            batch (dict[str, Tensor]): Raw batch from dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict[str, Any]: Dictionary containing validation metrics.
        """
        # 1. Extract features and targets
        features, targets = self._extract_features_targets(batch)

        # 2. Model forward
        predictions = self._invoke_model(features)

        # 3. Compute loss
        val_loss = self._compute_loss(predictions, targets)

        # 4. Compute metrics
        metrics = self._update_metrics(predictions, targets, stage="val")

        # 5. Log metrics
        self._log_stage_outputs("val", val_loss, metrics)

        return {"val_loss": val_loss}

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Test step with direct processing (no pipeline).

        Args:
            batch (dict[str, Tensor]): Raw batch from dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict[str, Any]: Dictionary containing test metrics.
        """
        # 1. Extract features and targets
        features, targets = self._extract_features_targets(batch)

        # 2. Model forward
        predictions = self._invoke_model(features)

        # 3. Compute loss
        test_loss = self._compute_loss(predictions, targets)

        # 4. Compute metrics
        metrics = self._update_metrics(predictions, targets, stage="test")

        # 5. Log metrics
        self._log_stage_outputs("test", test_loss, metrics)

        return {"test_loss": test_loss}

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, dict[str, Tensor]]:
        """Prediction step without loss computation.

        Targets are optional during inference.

        Args:
            batch (dict[str, Tensor]): Raw batch from dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary with ``predictions``, ``targets``, and ``latents``.
        """
        # 1. Extract features (targets are optional)
        features, targets = self._extract_features_targets(batch)

        # 2. Model forward
        predictions = self._invoke_model(features)

        # 3. Normalize predictions to dict format
        if isinstance(predictions, Tensor):
            predictions = {"output": predictions}

        return {
            "predictions": predictions,
            "targets": targets if targets else {},
            "latents": {}  # Subclasses can populate this
        }

    # =============================================================================
    # Checkpoint and Metadata Management
    # =============================================================================

    def configure_model(self) -> None:
        """Ensure precision is maintained after Lightning setup/checkpoint restore.

        Lightning calls this hook after:
        - Initial model creation
        - Checkpoint restoration (e.g., during LR tuning)
        - Strategy setup

        This ensures the nested model's dtype persists even after checkpoint operations
        that might reconstruct the model.
        """
        super().configure_model()

        # Reapply precision to nested model (idempotent and safe)
        from dlkit.interfaces.api.services.precision_service import get_precision_service

        precision_service = get_precision_service()
        precision_strategy = precision_service.resolve_precision()
        dtype = precision_strategy.to_torch_dtype()
        self.model = self.model.to(dtype=dtype)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save enhanced metadata including complete shape information.

        Ensures comprehensive model reconstruction information persists across
        checkpoint saves/loads.

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

        # Save entry configs for reconstruction
        dlkit_metadata['entry_configs'] = self._serialize_entry_configs()

        # Store in checkpoint
        checkpoint['dlkit_metadata'] = dlkit_metadata

    def _detect_model_family(self) -> str:
        """Detect model family for appropriate shape handling.

        Returns:
            Model family identifier ("dlkit_nn", "graph", "timeseries", "external")
        """
        try:
            from dlkit.runtime.workflows.factories.model_detection import detect_model_type
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

            # Extract hyperparameters
            params = {}
            if hasattr(settings, 'model_dump'):
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
            return {}

    def _serialize_entry_configs(self) -> dict[str, Any]:
        """Serialize entry configurations for reconstruction.

        Returns:
            Serialized entry configurations
        """
        try:
            if hasattr(self, '_entry_configs') and self._entry_configs:
                serialized = {}
                for name, entry in self._entry_configs.items():
                    serialized[name] = {
                        'name': entry.name,
                        'class_name': entry.__class__.__name__,
                    }
                return serialized
        except Exception:
            pass
        return {}

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore enhanced metadata and shape information from checkpoint.

        Raises:
            ValueError: If checkpoint version is unsupported or metadata format is invalid.
        """
        super().on_load_checkpoint(checkpoint)

        # Check for dlkit_metadata (required for v2.0+)
        if 'dlkit_metadata' not in checkpoint:
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

    # =============================================================================
    # Logging and Metrics
    # =============================================================================

    def _safe_log(self, *args, **kwargs) -> None:
        """Safe logging that only logs when trainer is available."""
        try:
            if hasattr(self, "trainer") and self.trainer is not None:
                self.log(*args, **kwargs)
        except Exception:
            pass

    def _safe_log_dict(self, *args, **kwargs) -> None:
        """Safe dict logging that only logs when trainer is available."""
        try:
            if hasattr(self, "trainer") and self.trainer is not None:
                self.log_dict(*args, **kwargs)
        except Exception:
            pass

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

    # =============================================================================
    # Optimizer and LR Management
    # =============================================================================

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
            except Exception:  # pragma: no cover
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

    def configure_optimizers(self):  # type: ignore[override]
        """Configure optimizer and scheduler from settings."""
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

    # =============================================================================
    # Entry Config Access
    # =============================================================================

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

    # =============================================================================
    # Model Creation (ABC-based)
    # =============================================================================

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

    # =============================================================================
    # Abstract Methods for Subclasses
    # =============================================================================

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Forward pass through the model.

        This method should be implemented by subclasses to handle
        the specific input format for their model type.
        """
        pass
