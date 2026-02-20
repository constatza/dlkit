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


class ProcessingLightningWrapper(LightningModule, ABC):
    """Abstract base Lightning wrapper with direct processing methods.

    Provides fundamental Lightning integration with streamlined processing:
    extraction → invocation → loss pairing → metrics.

    Transform functionality is provided by specialized subclasses (StandardLightningWrapper).

    Attributes:
        model (torch.nn.Module): Underlying PyTorch model.
        val_metrics (torchmetrics.MetricCollection): Validation metrics.
        test_metrics (torchmetrics.MetricCollection): Test metrics.
        loss_function (callable): Loss function operating on predictions and targets.
    """

    def __init__(
        self,
        *,
        settings: WrapperComponentSettings,
        model_settings: ModelComponentSettings,
        entry_configs: tuple[DataEntry, ...] | None = None,
        shape_summary: "ShapeSummary | None" = None,
        **kwargs,
    ):
        """Initialize the processing Lightning wrapper.

        Args:
            settings (WrapperComponentSettings): Wrapper configuration settings.
            model_settings (ModelComponentSettings): Model configuration settings.
            entry_configs (tuple[DataEntry, ...] | None): Data entry configurations.
            shape_summary: Shape summary from dataset inference (preferred).
            **kwargs: Additional arguments passed to LightningModule.
        """
        super().__init__()

        # Store configuration
        self._wrapper_settings = settings
        self._model_settings = model_settings
        self._entry_configs: tuple[DataEntry, ...] = entry_configs or ()

        self.save_hyperparameters(
            {
                "settings": settings,
                "model_settings": model_settings,
            },
            ignore=["settings", "model_settings", "entry_configs", "shape_summary"],
        )

        # Initialize model using build_model from core/models/factory.py
        from dlkit.core.models.factory import build_model
        from dlkit.core.shape_specs.simple_inference import ShapeSummary as _ShapeSummary
        import importlib

        # Extract model class
        if isinstance(model_settings.name, type):
            model_cls = model_settings.name
        else:
            module_path = getattr(model_settings, 'module_path', 'dlkit.core.models.nn')
            module = importlib.import_module(module_path)
            model_cls = getattr(module, model_settings.name)

        # Extract hyperparameters from model settings
        hyperparams = {}
        if hasattr(model_settings, 'model_dump'):
            all_fields = model_settings.model_dump()
            excluded = {'name', 'module_path', 'checkpoint'}
            hyperparams = {k: v for k, v in all_fields.items() if k not in excluded and v is not None}

        # Use shape_summary directly
        resolved_summary: _ShapeSummary | None = shape_summary

        self.model = build_model(model_cls, resolved_summary, hyperparams)
        self._shape_summary: _ShapeSummary | None = resolved_summary

        # Apply precision to model immediately after creation
        # Lightning's precision plugin will convert the wrapper, but we need to ensure
        # the nested model is also converted. This matches the pattern in test models.
        from dlkit.interfaces.api.services.precision_service import get_precision_service

        precision_service = get_precision_service()
        precision_strategy = precision_service.resolve_precision()
        dtype = precision_strategy.to_torch_dtype()
        self.model = self.model.to(dtype=dtype)

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

        # Ensure lr/learning_rate hyperparameters mirror optimizer settings
        self._sync_lr_hparam()

    # =============================================================================
    # Lightning Hooks
    # =============================================================================

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """Move batch tensors to device. Handles frozen Batch dataclass.

        Lightning's default implementation uses apply_to_collection which does not
        support frozen dataclasses. This override handles Batch explicitly.

        Args:
            batch: Incoming batch (Batch dataclass or other type).
            device: Target device.
            dataloader_idx: DataLoader index.

        Returns:
            Batch with all tensors moved to device.
        """
        from dataclasses import replace
        from dlkit.core.datatypes.batch import Batch as _Batch
        if isinstance(batch, _Batch):
            return replace(
                batch,
                features=tuple(f.to(device) for f in batch.features),
                targets=tuple(t.to(device) for t in batch.targets),
                latents=tuple(l.to(device) for l in batch.latents),
            )
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    # =============================================================================
    # Core Processing Helper Methods
    # =============================================================================

    def _log_dtype_mismatches(self, batch: "Batch") -> None:
        """Log dtype mismatches between first feature and model (debug only).

        Args:
            batch: Input batch with positional feature tensors.
        """
        if not batch.features or not hasattr(self.model, "dtype"):
            return
        model_dtype = self.model.dtype
        if not isinstance(model_dtype, torch.dtype):
            return
        feat = batch.features[0]
        if feat.is_floating_point() and feat.dtype != model_dtype:
            logger.debug(
                f"Feature dtype {feat.dtype} differs from model dtype {model_dtype}. "
                f"Lightning's precision plugin will handle alignment."
            )

    def _invoke_model(self, batch: "Batch") -> Tensor:
        """Dispatch positional Batch to model.

        Args:
            batch: Typed batch with positional feature tensors.

        Returns:
            Model output tensor.

        Raises:
            ValueError: If batch has no features.
        """
        match len(batch.features):
            case 0:
                raise ValueError("Batch has no features")
            case 1:
                return self.model(batch.features[0])
            case _:
                return self.model(*batch.features)

    def _compute_loss(self, predictions: Tensor, targets: tuple[Tensor, ...]) -> Tensor:
        """Compute loss positionally: pair predictions with targets[0], align dtype.

        Args:
            predictions: Model output tensor.
            targets: Tuple of target tensors (uses targets[0]).

        Returns:
            Scalar loss tensor.

        Raises:
            RuntimeError: If targets is empty.
        """
        if not targets:
            raise RuntimeError("Cannot compute loss: targets tuple is empty")
        target = targets[0]
        if target.is_floating_point() and target.dtype != predictions.dtype:
            target = target.to(dtype=predictions.dtype)
        return self.loss_function(predictions, target)

    def _update_metrics(
        self,
        predictions: Tensor,
        targets: tuple[Tensor, ...],
        stage: str,
    ) -> dict[str, Any]:
        """Compute metrics from predictions and first target.

        Args:
            predictions: Model output tensor.
            targets: Tuple of target tensors (uses targets[0]).
            stage: Stage identifier ("val" or "test").

        Returns:
            Dictionary of computed metric values, empty if no metrics or targets.
        """
        metrics = {"val": self.val_metrics, "test": self.test_metrics}.get(stage)
        if metrics is None or not targets:
            return {}
        target = targets[0]
        if target.is_floating_point() and target.dtype != predictions.dtype:
            target = target.to(dtype=predictions.dtype)
        return metrics(predictions, target)

    # =============================================================================
    # Lightning Step Methods
    # =============================================================================

    def training_step(self, batch: "Batch", batch_idx: int) -> dict[str, Any]:
        """Training step with direct positional processing.

        Args:
            batch: Positional batch from dataset.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing the training loss.
        """
        self._log_dtype_mismatches(batch)
        predictions = self._invoke_model(batch)
        loss = self._compute_loss(predictions, batch.targets)
        self._log_stage_outputs("train", loss)
        return {"loss": loss}

    def validation_step(self, batch: "Batch", batch_idx: int) -> dict[str, Any]:
        """Validation step with direct positional processing.

        Args:
            batch: Positional batch from dataset.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing validation metrics.
        """
        self._log_dtype_mismatches(batch)
        predictions = self._invoke_model(batch)
        val_loss = self._compute_loss(predictions, batch.targets)
        metrics = self._update_metrics(predictions, batch.targets, "val")
        self._log_stage_outputs("val", val_loss, metrics)
        return {"val_loss": val_loss}

    def test_step(self, batch: "Batch", batch_idx: int) -> dict[str, Any]:
        """Test step with direct positional processing.

        Args:
            batch: Positional batch from dataset.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing test metrics.
        """
        self._log_dtype_mismatches(batch)
        predictions = self._invoke_model(batch)
        test_loss = self._compute_loss(predictions, batch.targets)
        metrics = self._update_metrics(predictions, batch.targets, "test")
        self._log_stage_outputs("test", test_loss, metrics)
        return {"test_loss": test_loss}

    def predict_step(self, batch: "Batch", batch_idx: int) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Prediction step without loss computation.

        Args:
            batch: Positional batch from dataset.
            batch_idx: Index of the batch.

        Returns:
            Tuple of (predictions, targets, latents), each containing a tuple of tensors.
        """
        predictions = self._invoke_model(batch)
        return (
            (predictions,) if isinstance(predictions, Tensor) else predictions,
            batch.targets,
            batch.latents,
        )

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

        # Save model settings for reconstruction
        dlkit_metadata['model_settings'] = self._serialize_model_settings()

        # Save entry configs for reconstruction
        dlkit_metadata['entry_configs'] = self._serialize_entry_configs()

        # Save shape summary for model input/output documentation
        dlkit_metadata['shape_summary'] = self._compute_shape_summary()

        # Store in checkpoint
        checkpoint['dlkit_metadata'] = dlkit_metadata

    def _detect_model_family(self) -> str:
        """Detect model family for appropriate shape handling.

        Returns:
            Model family identifier ("dlkit_nn", "graph", "timeseries", "external")
        """
        try:
            from dlkit.runtime.workflows.factories.model_detection import detect_model_type
            model_type = detect_model_type(self._model_settings, None)  # type: ignore[arg-type]
            return model_type.value
        except Exception:
            return "external"

    def _serialize_model_settings(self) -> dict[str, Any]:
        """Serialize model settings for reconstruction.

        Returns:
            Serialized model configuration
        """
        try:
            settings = self._model_settings

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

    def _serialize_entry_configs(self) -> list[dict[str, Any]]:
        """Serialize entry configurations for reconstruction.

        Returns:
            List of serialized entry configurations
        """
        try:
            if hasattr(self, '_entry_configs') and self._entry_configs:
                return [
                    {
                        "name": e.name,
                        "class_name": e.__class__.__name__,
                        "transforms": [
                            t.model_dump() if hasattr(t, "model_dump") else t
                            for t in getattr(e, "transforms", [])
                        ]
                    }
                    for e in self._entry_configs
                ]
        except Exception:
            pass
        return []

    def _compute_shape_summary(self) -> dict[str, Any]:
        """Serialize stored ShapeSummary for checkpoint persistence.

        Returns:
            Dict with in_shapes and out_shapes lists, or empty dict if unavailable.
        """
        if not hasattr(self, '_shape_summary') or self._shape_summary is None:
            return {}
        return {
            'in_shapes': [list(s) for s in self._shape_summary.in_shapes],
            'out_shapes': [list(s) for s in self._shape_summary.out_shapes],
        }

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

    def get_entry_configs(self) -> tuple[DataEntry, ...]:
        """Get the data entry configurations used by this wrapper.

        Returns:
            Tuple of DataEntry configurations
        """
        return self._entry_configs

    def get_feature_configs(self) -> tuple[DataEntry, ...]:
        """Get feature entry configurations.

        Returns:
            Tuple of Feature configurations
        """
        return tuple(e for e in self._entry_configs if is_feature_entry(e))

    def get_target_configs(self) -> tuple[DataEntry, ...]:
        """Get target entry configurations.

        Returns:
            Tuple of Target configurations
        """
        return tuple(e for e in self._entry_configs if is_target_entry(e))


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
