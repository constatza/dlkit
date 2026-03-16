"""Base Lightning wrapper with protocol-based composition.

This module defines the abstract base class for all Lightning wrappers.
All computation is delegated to injected SOLID protocol objects; the wrapper
is a pure Lightning coordinator.
"""

from abc import abstractmethod
from collections.abc import Sequence
from functools import reduce
from typing import Any, cast

from loguru import logger

# Configure checkpoint loading for PyTorch 2.6+ to allow Pydantic settings
from dlkit.core.models.wrappers.security import configure_checkpoint_loading

configure_checkpoint_loading()

import torch
import torch.nn as nn
from lightning import LightningModule
from tensordict import TensorDict
from torch import Tensor

from dlkit.tools.config import BuildContext, FactoryProvider
from dlkit.tools.config.core.updater import update_settings
from dlkit.core.models.wrappers.protocols import (
    IModelInvoker,
    ILossComputer,
    IMetricsUpdater,
    IBatchTransformer,
    IFittableBatchTransformer,
    IBatchTransform,
    IGeneratorFactory,
    IPredictionStrategy,
)
from dlkit.core.models.wrappers.components import WrapperCheckpointMetadata


def _unpack_model_output(raw_output: Any) -> tuple[Any, Any]:
    """Extract (predictions_raw, latents_raw) from any model forward() output.

    Handles structural unpacking only: list rejection, tuple splitting, and
    self-describing dicts/TensorDicts that carry a ``"predictions"`` key.

    Args:
        raw_output: Raw output from the model forward pass.

    Returns:
        Tuple of ``(predictions_raw, latents_raw)``. ``latents_raw`` is
        ``None`` when absent.

    Raises:
        TypeError: If raw_output is a list (ambiguous at top level).
        ValueError: If raw_output is an empty tuple.
    """
    if isinstance(raw_output, list):
        raise TypeError(
            "forward() returned a list — ambiguous at top level. "
            "Use dict[str, ...] for multi-head or a (predictions, latents) tuple."
        )
    if isinstance(raw_output, tuple):
        match len(raw_output):
            case 0:
                raise ValueError("forward() returned an empty tuple")
            case 1:
                return raw_output[0], None
            case 2:
                return raw_output[0], raw_output[1]
            case _:
                return raw_output[0], raw_output[1:]  # latents = tuple → positional TD
    # Self-describing: dict or TensorDict with "predictions" key
    if isinstance(raw_output, (dict, TensorDict)) and "predictions" in raw_output:
        latents_raw = (
            raw_output.get("latents", None)
            if isinstance(raw_output, TensorDict)
            else raw_output.get("latents")
        )
        return raw_output["predictions"], latents_raw
    return raw_output, None


def _batch_size_of(value: Any) -> int:
    """Infer batch size from the first dimension of a Tensor, TensorDict, or dict.

    Args:
        value: A Tensor, TensorDict, or non-empty dict with Tensor/TensorDict values.

    Returns:
        Integer batch size.

    Raises:
        ValueError: If value is an empty dict.
        TypeError: If value type is unsupported.
    """
    match value:
        case torch.Tensor():
            return value.shape[0]
        case TensorDict():
            return int(value.batch_size[0])
        case dict() if value:
            return _batch_size_of(next(iter(value.values())))
        case dict():
            raise ValueError("Cannot determine batch size from empty dict")
        case _:
            raise TypeError(f"Cannot determine batch size from {type(value).__name__}")


# Maximum nesting depth for recursive model output normalization.
# Each dict/sequence level consumed from forward() increments the counter;
# Tensor/TensorDict leaves terminate immediately. A depth of 8 supports any
# realistic multi-head/latent structure while bounding runaway recursion.
_MAX_NORMALIZE_DEPTH = 8


def _normalize_dict(d: dict[str, Any], context: str, batch_size: int, _depth: int) -> TensorDict:
    """Normalize a plain dict to TensorDict with a per-entry leaf fast-path.

    Args:
        d: Source dict mapping string keys to output values.
        context: Human-readable context string for error messages.
        batch_size: Batch dimension to set on the resulting TensorDict.
        _depth: Current recursion depth (internal — callers pass 0 at top level).

    Returns:
        TensorDict with ``batch_size=[batch_size]``.
    """
    normalized: dict[str, Tensor | TensorDict] = {}
    for k, v in d.items():
        match v:
            case torch.Tensor() | TensorDict():
                normalized[k] = v
            case _:
                normalized[k] = _normalize_output(v, f"{context}.{k}", batch_size, _depth)
    return TensorDict(normalized, batch_size=[batch_size])


def _normalize_sequence(
    seq: list | tuple, context: str, batch_size: int, _depth: int
) -> TensorDict:
    """Normalize a positional sequence to TensorDict with integer string keys.

    Args:
        seq: Non-empty list or tuple of output values.
        context: Human-readable context string for error messages.
        batch_size: Batch dimension to set on the resulting TensorDict.
        _depth: Current recursion depth (internal — callers pass 0 at top level).

    Returns:
        TensorDict with keys ``"0"``, ``"1"``, … and ``batch_size=[batch_size]``.
    """
    from dlkit.tools.utils.tensordict_utils import sequence_to_tensordict

    normalized = [
        _normalize_output(v, f"{context}[{i}]", batch_size, _depth) for i, v in enumerate(seq)
    ]
    return sequence_to_tensordict(normalized)


def _normalize_output(
    value: Any, context: str, batch_size: int, _depth: int = 0
) -> Tensor | TensorDict:
    """Recursively normalize a model output value to Tensor or TensorDict.

    Each recursive call increments *_depth*; a ``RecursionError`` is raised
    when ``_depth`` exceeds ``_MAX_NORMALIZE_DEPTH``, preventing infinite loops
    from pathological or circular model outputs.

    Args:
        value: A Tensor, TensorDict, dict, list, or tuple.
        context: Human-readable context string for error messages.
        batch_size: Batch dimension used when constructing nested TensorDicts.
        _depth: Current recursion depth; callers should omit (defaults to 0).

    Returns:
        Tensor (unchanged) or TensorDict (constructed).

    Raises:
        RecursionError: If nesting depth exceeds ``_MAX_NORMALIZE_DEPTH``.
        ValueError: If value is an empty dict, list, or tuple.
        TypeError: If value type is unsupported.
    """
    if _depth > _MAX_NORMALIZE_DEPTH:
        raise RecursionError(
            f"{context}: model output nesting exceeds maximum depth "
            f"{_MAX_NORMALIZE_DEPTH}. Ensure forward() returns Tensors at the leaves."
        )
    match value:
        case torch.Tensor():
            return value
        case TensorDict():
            return value
        case dict() if value:
            return _normalize_dict(value, context, batch_size, _depth + 1)
        case dict():
            raise ValueError(f"{context}: empty dict is not a valid model output")
        case list() | tuple() if value:
            return _normalize_sequence(value, context, batch_size, _depth + 1)
        case list() | tuple():
            raise ValueError(f"{context}: empty sequence is not a valid model output")
        case _:
            raise TypeError(
                f"{context}: unsupported output type {type(value).__name__}. "
                "Expected Tensor, TensorDict, dict, list, or tuple."
            )


def _leaf_dtype(value: Tensor | TensorDict) -> torch.dtype:
    """Return the dtype of the first leaf Tensor in value.

    Falls back to ``torch.float32`` when *value* is an empty TensorDict.

    Args:
        value: A Tensor or (possibly nested) TensorDict.

    Returns:
        The ``torch.dtype`` of the first leaf tensor found, or
        ``torch.float32`` when *value* contains no tensors.
    """
    v: Any = value
    while isinstance(v, TensorDict):
        try:
            v = next(iter(v.values()))
        except StopIteration:
            return torch.float32
    return cast(Tensor, v).dtype


def _leaf_device(value: Tensor | TensorDict) -> torch.device:
    """Return the device of the first leaf Tensor in value.

    Falls back to ``torch.device("cpu")`` when *value* is an empty TensorDict.

    Args:
        value: A Tensor or (possibly nested) TensorDict.

    Returns:
        The ``torch.device`` of the first leaf tensor found, or
        ``torch.device("cpu")`` when *value* contains no tensors.
    """
    v: Any = value
    while isinstance(v, TensorDict):
        try:
            v = next(iter(v.values()))
        except StopIteration:
            return torch.device("cpu")
    return cast(Tensor, v).device


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
        prediction_strategy: IPredictionStrategy,
        batch_transforms: Sequence[IBatchTransform] = (),
        train_generator_factory: IGeneratorFactory | None = None,
        val_generator_factory: IGeneratorFactory | None = None,
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
            prediction_strategy: Strategy that implements predict_step logic.
                Standard models use ``DiscriminativePredictionStrategy``; generative
                models use ``ODEPredictionStrategy``.
            batch_transforms: Sequence of coupled supervision transforms applied per batch
                before the per-slot batch_transformer chains. Empty by default (no-op).
            train_generator_factory: Produces a generator per training batch for
                reproducible stochastic transforms. Defaults to NullGeneratorFactory (global RNG).
            val_generator_factory: Produces a generator per val/test batch. Defaults to
                NullGeneratorFactory (global RNG).
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
        self._prediction_strategy = prediction_strategy
        self._batch_transforms: tuple[IBatchTransform, ...] = tuple(batch_transforms)

        from dlkit.core.models.wrappers.generator_factories import NullGeneratorFactory

        _null = NullGeneratorFactory()
        self._train_generator_factory: IGeneratorFactory = train_generator_factory or _null
        self._val_generator_factory: IGeneratorFactory = val_generator_factory or _null

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
        from loguru import logger

        loader = dm.train_dataloader()
        logger.info("Starting transform fitting from training dataloader.")
        self._batch_transformer.fit(loader)
        logger.info("Finished transform fitting.")

    # =========================================================================
    # Batch Transform Helpers
    # =========================================================================

    def _apply_batch_transforms(
        self,
        batch: Any,
        generator: "torch.Generator | None",
    ) -> Any:
        """Apply coupled supervision transforms in sequence.

        Each ``IBatchTransform`` in ``self._batch_transforms`` is called with
        the batch and generator, the result piped to the next transform via
        ``reduce``.  An empty sequence is a no-op.

        Args:
            batch: Input TensorDict.
            generator: Optional RNG for stochastic transforms.

        Returns:
            Transformed TensorDict (identity when no transforms configured).
        """
        if not self._batch_transforms:
            return batch
        return reduce(lambda b, t: t(b, generator), self._batch_transforms, batch)

    def _compute_loss(self, predictions: Any, batch: Any) -> "torch.Tensor":
        """Compute scalar loss from predictions and batch.

        Delegates to ``self._loss_computer`` by default. Subclasses can
        override this method to implement custom loss logic (e.g.
        ``FlowMatchingWrapper`` targets ``batch["targets"]["ut"]`` directly).

        Args:
            predictions: Model output tensor.
            batch: Enriched TensorDict with features, targets, and predictions.

        Returns:
            Scalar loss tensor.
        """
        return self._loss_computer.compute(predictions, batch)

    # =========================================================================
    # Lightning Step Methods (delegate to protocols)
    # =========================================================================

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        """Training step: coupled-transforms → chain-transforms → invoke → loss → log.

        Applies coupled supervision transforms (``_batch_transforms``) first,
        then per-slot normalisation chains (``_batch_transformer``), then
        invokes the model and computes the loss.

        Args:
            batch: TensorDict batch from dataset.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing the training loss.
        """
        gen = self._train_generator_factory(batch_idx)
        batch = self._apply_batch_transforms(batch, gen)
        batch = self._batch_transformer.transform(batch)
        batch = self._model_invoker.invoke(self.model, batch)
        loss = self._compute_loss(batch["predictions"], batch)
        batch_size = _batch_size_of(batch["predictions"])
        self._log_stage_outputs("train", loss, batch_size=batch_size)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        """Validation step: coupled-transforms → chain-transforms → invoke → loss → metrics → log.

        Args:
            batch: TensorDict batch from dataset.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing validation loss.
        """
        gen = self._val_generator_factory(batch_idx)
        batch = self._apply_batch_transforms(batch, gen)
        batch = self._batch_transformer.transform(batch)
        batch = self._model_invoker.invoke(self.model, batch)
        val_loss = self._compute_loss(batch["predictions"], batch)
        batch_size = _batch_size_of(batch["predictions"])
        self._metrics_updater.update(batch["predictions"], batch, stage="val")
        self._log_stage_outputs("val", val_loss, batch_size=batch_size)
        return {"val_loss": val_loss}

    def test_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        """Test step: coupled-transforms → chain-transforms → invoke → loss → metrics → log.

        Args:
            batch: TensorDict batch from dataset.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing test loss.
        """
        gen = self._val_generator_factory(batch_idx)
        batch = self._apply_batch_transforms(batch, gen)
        batch = self._batch_transformer.transform(batch)
        batch = self._model_invoker.invoke(self.model, batch)
        test_loss = self._compute_loss(batch["predictions"], batch)
        batch_size = _batch_size_of(batch["predictions"])
        self._metrics_updater.update(batch["predictions"], batch, stage="test")
        self._log_stage_outputs("test", test_loss, batch_size=batch_size)
        return {"test_loss": test_loss}

    def predict_step(self, batch: Any, batch_idx: int) -> TensorDict:
        """Prediction step returning a TensorDict with predictions, targets and latents.

        When ``_prediction_strategy`` is set (e.g. for generative models), the
        entire predict logic is delegated to the strategy object.  Otherwise
        the legacy discriminative path is used (backward compatible).

        Args:
            batch: TensorDict batch from dataset.
            batch_idx: Index of the batch.

        Returns:
            TensorDict with keys ``"predictions"``, ``"targets"``, and
            ``"latents"`` (zero-size ``(B, 0)`` sentinel when absent).
        """
        gen = self._val_generator_factory(batch_idx)
        return self._prediction_strategy.predict(self.model, batch, gen)

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
            dlkit_metadata["model_settings"] = self._serialize_model_settings(meta.model_settings)
            dlkit_metadata["entry_configs"] = self._serialize_entry_configs(meta.entry_configs)
            dlkit_metadata["shape_summary"] = self._compute_shape_summary(meta.shape_summary)
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
                    self._checkpoint_metadata.model_settings,
                    None,  # type: ignore[arg-type]
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
                    k: v for k, v in all_fields.items() if k not in excluded and v is not None
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
        batch_size: int | None = None,
    ) -> None:
        """Centralized metric logging for training stages.

        Args:
            stage: Stage identifier ('train', 'val', 'test', 'val_epoch', 'test_epoch').
            loss: Scalar loss tensor (optional).
            metrics: Additional metrics dict (optional).
            batch_size: Batch size for correct epoch-level weighted averaging by Lightning.
        """
        if loss is not None:
            loss_name = self._format_metric_name(stage, "loss")
            self._safe_log(
                loss_name, loss, on_step=False, on_epoch=True, prog_bar=True,
                batch_size=batch_size,
            )
        if metrics:
            formatted = {
                self._format_metric_name(stage, key): value for key, value in metrics.items()
            }
            self._safe_log_dict(
                formatted, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size,
            )

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
