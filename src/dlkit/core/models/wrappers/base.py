"""Base Lightning wrapper with protocol-based composition.

This module defines the abstract base class for all Lightning wrappers.
All computation is delegated to injected SOLID protocol objects; the wrapper
is a pure Lightning coordinator.
"""

from abc import abstractmethod
from typing import Any, cast

from loguru import logger

# Configure checkpoint loading for PyTorch 2.6+ to allow Pydantic settings
from dlkit.tools.utils.checkpoint_security import configure_checkpoint_loading
configure_checkpoint_loading()

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from dlkit.tools.config import BuildContext, FactoryProvider
from dlkit.tools.config.core.updater import update_settings
from dlkit.core.models.wrappers.protocols import (
    IModelInvoker,
    ILossComputer,
    IMetricsUpdater,
    IBatchTransformer,
    IFittableBatchTransformer,
)
from dlkit.core.models.wrappers.components import WrapperCheckpointMetadata


def _unpack_model_output(raw_output: Any) -> tuple[Tensor, Tensor | None]:
    """Split model output into (predictions, latents | None).

    Models returning a 2-tuple are assumed to emit (predictions, latents).
    All other outputs are treated as predictions-only.

    Args:
        raw_output: Raw output from the model forward pass.

    Returns:
        Tuple of (predictions tensor, latents tensor or None).
    """
    if isinstance(raw_output, tuple) and len(raw_output) == 2:
        return cast(Tensor, raw_output[0]), cast(Tensor | None, raw_output[1])
    return cast(Tensor, raw_output), None


def _build_predict_tensordict(
    predictions: Tensor,
    targets: Any,
    latents: Tensor | None = None,
) -> Any:
    """Assemble the predict_step TensorDict from its components.

    Args:
        predictions: Inverse-transformed prediction tensor.
        targets: Raw (untransformed) targets from the batch.
        latents: Optional latent tensor from the model.

    Returns:
        TensorDict with keys 'predictions', 'targets', and optionally 'latents'.
    """
    from tensordict import TensorDict
    contents: dict[str, Any] = {"predictions": predictions, "targets": targets}
    if latents is not None:
        contents["latents"] = latents
    return TensorDict(contents, batch_size=predictions.shape[:1])


def _build_model_from_settings(model_settings: Any, shape_summary: Any = None) -> nn.Module:
    """Build a PyTorch model from configuration settings.

    Args:
        model_settings: Model configuration (ModelComponentSettings).
        shape_summary: Optional ShapeSummary for shape-aware models.

    Returns:
        Instantiated and precision-cast nn.Module.
    """
    import importlib
    from dlkit.core.models.factory import build_model

    if isinstance(model_settings.name, type):
        model_cls = model_settings.name
    else:
        module_path = getattr(model_settings, "module_path", "dlkit.core.models.nn")
        module = importlib.import_module(module_path)
        model_cls = getattr(module, model_settings.name)

    hyperparams: dict[str, Any] = {}
    if hasattr(model_settings, "model_dump"):
        all_fields = model_settings.model_dump()
        excluded = {"name", "module_path", "checkpoint"}
        hyperparams = {k: v for k, v in all_fields.items() if k not in excluded and v is not None}

    return build_model(model_cls, shape_summary, hyperparams)


class ProcessingLightningWrapper(LightningModule):
    """Pure Lightning coordinator. All computation delegated to injected protocols.

    Accepts pre-built model and protocol objects; delegates every computation
    step to them. No model building, no direct transform application, no
    positional batch assumptions live here.

    Attributes:
        model: Underlying PyTorch model.
        optimizer: Optimizer settings (OptimizerSettings).
        scheduler: Scheduler settings (SchedulerSettings | None).
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        model_invoker: IModelInvoker,
        loss_computer: ILossComputer,
        metrics_updater: IMetricsUpdater,
        batch_transformer: IBatchTransformer,
        optimizer_settings: Any,
        scheduler_settings: Any = None,
        predict_target_key: str,
        checkpoint_metadata: WrapperCheckpointMetadata | None = None,
    ) -> None:
        """Initialize the wrapper with injected protocol objects.

        Args:
            model: Pre-built PyTorch nn.Module.
            model_invoker: Extracts features and invokes model.
            loss_computer: Computes scalar loss from predictions + batch.
            metrics_updater: Accumulates and exposes metric state.
            batch_transformer: Applies transforms to batches (nn.Module for state persistence).
            optimizer_settings: Optimizer configuration settings.
            scheduler_settings: Scheduler configuration settings (optional).
            predict_target_key: Target entry name whose chain is inverted at predict time.
            checkpoint_metadata: Serialisation-only metadata for checkpoint persistence.
        """
        super().__init__()

        self.model = model
        self._model_invoker = model_invoker
        self._loss_computer = loss_computer
        self._metrics_updater = metrics_updater
        self._batch_transformer = batch_transformer
        self.optimizer = optimizer_settings
        self.scheduler = scheduler_settings
        self._predict_target_key = predict_target_key
        self._checkpoint_metadata = checkpoint_metadata

        # Entry configs for subclass access (e.g. graph wrapper target name)
        self._entry_configs: tuple[Any, ...] = (
            checkpoint_metadata.entry_configs if checkpoint_metadata is not None else ()
        )

        # Apply precision to model immediately after creation
        from dlkit.interfaces.api.services.precision_service import get_precision_service
        precision_service = get_precision_service()
        precision_strategy = precision_service.resolve_precision()
        dtype = precision_strategy.to_torch_dtype()
        self.model = self.model.to(dtype=dtype)

        self._sync_lr_hparam()

    # =========================================================================
    # Lightning Hooks
    # =========================================================================

    def configure_model(self) -> None:
        """Reapply precision after Lightning setup/checkpoint restore."""
        super().configure_model()
        from dlkit.interfaces.api.services.precision_service import get_precision_service
        precision_service = get_precision_service()
        precision_strategy = precision_service.resolve_precision()
        dtype = precision_strategy.to_torch_dtype()
        self.model = self.model.to(dtype=dtype)

    def on_fit_start(self) -> None:
        """Fit the batch transformer if it implements IFittableBatchTransformer."""
        if not isinstance(self._batch_transformer, IFittableBatchTransformer):
            return
        if self._batch_transformer.is_fitted():
            return
        trainer = getattr(self, "trainer", None)
        if trainer is None or not hasattr(trainer, "datamodule"):
            return
        dm = trainer.datamodule
        if dm is None or not hasattr(dm, "train_dataloader"):
            return
        try:
            loader = dm.train_dataloader()
            self._batch_transformer.fit(loader)
        except Exception:
            pass

    # =========================================================================
    # Lightning Step Methods (delegate to protocols)
    # =========================================================================

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        """Training step: transform → invoke → loss → log.

        Args:
            batch: TensorDict batch from dataset.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing the training loss.
        """
        batch = self._batch_transformer.transform(batch)
        predictions = self._model_invoker.invoke(self.model, batch)
        loss = self._loss_computer.compute(predictions, batch)
        self._log_stage_outputs("train", loss)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        """Validation step: transform → invoke → loss → metrics → log.

        Args:
            batch: TensorDict batch from dataset.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing validation loss.
        """
        batch = self._batch_transformer.transform(batch)
        predictions = self._model_invoker.invoke(self.model, batch)
        val_loss = self._loss_computer.compute(predictions, batch)
        self._metrics_updater.update(predictions, batch, stage="val")
        self._log_stage_outputs("val", val_loss)
        return {"val_loss": val_loss}

    def test_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        """Test step: transform → invoke → loss → metrics → log.

        Args:
            batch: TensorDict batch from dataset.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing test loss.
        """
        batch = self._batch_transformer.transform(batch)
        predictions = self._model_invoker.invoke(self.model, batch)
        test_loss = self._loss_computer.compute(predictions, batch)
        self._metrics_updater.update(predictions, batch, stage="test")
        self._log_stage_outputs("test", test_loss)
        return {"test_loss": test_loss}

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        """Prediction step returning a TensorDict with predictions and targets.

        Captures original (untransformed) targets before applying the batch
        transform, so predictions and targets are guaranteed to be aligned.
        If the model returns a 2-tuple ``(predictions, latents)``, the latents
        are included in the output under the ``"latents"`` key.

        Args:
            batch: TensorDict batch from dataset.
            batch_idx: Index of the batch.

        Returns:
            TensorDict with keys ``"predictions"``, ``"targets"``, and
            optionally ``"latents"``.
        """
        original_targets = batch["targets"]
        batch = self._batch_transformer.transform(batch)
        raw_output = self._model_invoker.invoke(self.model, batch)
        predictions, latents = _unpack_model_output(raw_output)
        predictions = self._batch_transformer.inverse_transform_predictions(
            predictions, self._predict_target_key
        )
        return _build_predict_tensordict(predictions, original_targets, latents)

    @staticmethod
    def collect_targets(predict_outputs: list[Any]) -> list[Any]:
        """Extract targets from ``trainer.predict()`` outputs.

        Since ``predict_step`` embeds targets inside each output TensorDict,
        targets are always aligned with predictions — no second dataloader pass
        needed.

        Args:
            predict_outputs: List returned by ``trainer.predict()``, where each
                element is a TensorDict produced by ``predict_step``.

        Returns:
            List of per-batch target TensorDicts (one entry per batch).
        """
        return [batch["targets"] for batch in predict_outputs]

    def on_validation_epoch_end(self) -> None:
        """Compute and log epoch-level validation metrics, then reset."""
        metrics = self._metrics_updater.compute("val")
        if metrics:
            self._log_stage_outputs("val_epoch", None, metrics)
        self._metrics_updater.reset("val")

    def on_test_epoch_end(self) -> None:
        """Compute and log epoch-level test metrics, then reset."""
        metrics = self._metrics_updater.compute("test")
        if metrics:
            self._log_stage_outputs("test_epoch", None, metrics)
        self._metrics_updater.reset("test")

    # =========================================================================
    # Checkpoint and Metadata Management
    # =========================================================================

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save DLKit metadata to checkpoint.

        Args:
            checkpoint: Checkpoint dict to augment.
        """
        super().on_save_checkpoint(checkpoint)

        dlkit_metadata: dict[str, Any] = {
            "version": "2.0",
            "wrapper_type": self.__class__.__name__,
        }

        if self._checkpoint_metadata is not None:
            meta = self._checkpoint_metadata
            dlkit_metadata["model_settings"] = self._serialize_model_settings(
                meta.model_settings
            )
            dlkit_metadata["entry_configs"] = self._serialize_entry_configs(
                meta.entry_configs
            )
            dlkit_metadata["shape_summary"] = self._compute_shape_summary(
                meta.shape_summary
            )
            dlkit_metadata["feature_names"] = list(meta.feature_names)
            dlkit_metadata["predict_target_key"] = meta.predict_target_key
            dlkit_metadata["model_family"] = self._detect_model_family()
        else:
            dlkit_metadata["model_settings"] = {}
            dlkit_metadata["entry_configs"] = []
            dlkit_metadata["shape_summary"] = {}
            dlkit_metadata["model_family"] = "external"

        checkpoint["dlkit_metadata"] = dlkit_metadata

    def _detect_model_family(self) -> str:
        """Detect model family identifier.

        Returns:
            Model family string (e.g., 'dlkit_nn', 'graph', 'external').
        """
        try:
            from dlkit.runtime.workflows.factories.model_detection import detect_model_type
            if self._checkpoint_metadata is not None:
                model_type = detect_model_type(
                    self._checkpoint_metadata.model_settings, None  # type: ignore[arg-type]
                )
                return model_type.value
        except Exception:
            pass
        return "external"

    def _serialize_model_settings(self, model_settings: Any) -> dict[str, Any]:
        """Serialize model settings for checkpoint reconstruction.

        Args:
            model_settings: Model configuration settings.

        Returns:
            Serialized model configuration dict.
        """
        try:
            name = getattr(model_settings, "name", None)
            module_path = getattr(model_settings, "module_path", None)
            class_name = model_settings.__class__.__name__
            params: dict[str, Any] = {}
            if hasattr(model_settings, "model_dump"):
                all_fields = model_settings.model_dump()
                excluded = {"name", "module_path", "checkpoint"}
                params = {
                    k: v
                    for k, v in all_fields.items()
                    if k not in excluded and v is not None
                }
            return {
                "name": name,
                "module_path": module_path,
                "params": params,
                "class_name": class_name,
            }
        except Exception:
            return {}

    def _serialize_entry_configs(self, entry_configs: tuple) -> list[dict[str, Any]]:
        """Serialize entry configurations.

        Args:
            entry_configs: Tuple of DataEntry objects.

        Returns:
            List of serialized entry config dicts.
        """
        try:
            return [
                {
                    "name": e.name,
                    "class_name": e.__class__.__name__,
                    "transforms": [
                        t.model_dump() if hasattr(t, "model_dump") else t
                        for t in getattr(e, "transforms", [])
                    ],
                }
                for e in entry_configs
            ]
        except Exception:
            return []

    def _compute_shape_summary(self, shape_summary: Any) -> dict[str, Any]:
        """Serialize ShapeSummary for checkpoint persistence.

        Args:
            shape_summary: ShapeSummary instance or None.

        Returns:
            Dict with in_shapes/out_shapes or empty dict.
        """
        if shape_summary is None:
            return {}
        try:
            return {
                "in_shapes": [list(s) for s in shape_summary.in_shapes],
                "out_shapes": [list(s) for s in shape_summary.out_shapes],
            }
        except Exception:
            return {}

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore and validate checkpoint metadata.

        Args:
            checkpoint: Checkpoint dict to restore from.

        Raises:
            ValueError: If checkpoint version is unsupported or metadata is missing.
        """
        super().on_load_checkpoint(checkpoint)

        if "dlkit_metadata" not in checkpoint:
            raise ValueError(
                "Checkpoint missing 'dlkit_metadata'. This checkpoint uses a legacy format "
                "that is no longer supported. Please re-train your model with the current "
                "version of dlkit to generate a compatible checkpoint."
            )

        metadata = checkpoint["dlkit_metadata"]
        version = metadata.get("version")
        if version is None:
            raise ValueError(
                "Checkpoint metadata missing 'version' field. Cannot verify compatibility."
            )
        if version != "2.0":
            raise ValueError(
                f"Unsupported checkpoint version '{version}'. Only version '2.0' is supported."
            )

    # =========================================================================
    # Logging Helpers
    # =========================================================================

    def _safe_log(self, *args, **kwargs) -> None:
        """Log safely when trainer is available."""
        try:
            if hasattr(self, "trainer") and self.trainer is not None:
                self.log(*args, **kwargs)
        except Exception:
            pass

    def _safe_log_dict(self, *args, **kwargs) -> None:
        """Log dict safely when trainer is available."""
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
        """Centralized metric logging for training stages.

        Args:
            stage: Stage identifier ('train', 'val', 'test', 'val_epoch', 'test_epoch').
            loss: Scalar loss tensor (optional).
            metrics: Additional metrics dict (optional).
        """
        if loss is not None:
            loss_name = self._format_metric_name(stage, "loss")
            self._safe_log(loss_name, loss, on_step=False, on_epoch=True, prog_bar=True)
        if metrics:
            formatted = {
                self._format_metric_name(stage, key): value
                for key, value in metrics.items()
            }
            self._safe_log_dict(formatted, on_step=False, on_epoch=True, prog_bar=True)

    def _format_metric_name(self, stage: str, name: str) -> str:
        """Normalize metric names per stage conventions.

        Args:
            stage: Stage identifier.
            name: Raw metric name.

        Returns:
            Normalized metric name string.
        """
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

        return f"{stage_lower}_{name}"

    def on_train_epoch_end(self) -> None:
        """Log current learning rate at epoch end."""
        if self.trainer and self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self._safe_log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)

    # =========================================================================
    # Optimizer and LR Management
    # =========================================================================

    def _get_attached_trainer(self) -> Any:
        """Return attached trainer without triggering Lightning errors."""
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            return trainer
        fabric = getattr(self, "_fabric", None)
        if fabric is not None:
            try:
                from lightning.pytorch.core.module import _TrainerFabricShim
                return _TrainerFabricShim(fabric=fabric)
            except Exception:
                return None
        return None

    def _get_optimizer_lr(self) -> float | None:
        """Resolve effective learning rate from optimizer settings or trainer.

        Returns:
            Current learning rate as float, or None if unavailable.
        """
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
        """Synchronise stored hyperparameters with the current learning rate."""
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
        """Alias for lr, expected by some Lightning utilities."""
        return self.lr

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.lr = value

    def configure_optimizers(self):  # type: ignore[override]
        """Configure optimizer and scheduler from stored settings."""
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

    # =========================================================================
    # Entry Config Access
    # =========================================================================

    def get_entry_configs(self) -> tuple[Any, ...]:
        """Get the data entry configurations used by this wrapper.

        Returns:
            Tuple of DataEntry configurations.
        """
        return self._entry_configs

    def get_feature_configs(self) -> tuple[Any, ...]:
        """Get feature entry configurations.

        Returns:
            Tuple of Feature configurations.
        """
        from dlkit.tools.config.data_entries import is_feature_entry
        return tuple(e for e in self._entry_configs if is_feature_entry(e))

    def get_target_configs(self) -> tuple[Any, ...]:
        """Get target entry configurations.

        Returns:
            Tuple of Target configurations.
        """
        from dlkit.tools.config.data_entries import is_target_entry
        return tuple(e for e in self._entry_configs if is_target_entry(e))

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Forward pass through the model.

        Subclasses implement the specific input format for their model type.
        """
