"""Lightning callbacks for wrapper lifecycle concerns.

Provides reusable Lightning Callbacks that replace lifecycle methods
previously embedded in the wrapper classes:
- TransformFittingCallback: Fits NamedBatchTransformer before training starts.
- MLflowEpochLogger: Logs epoch-based metrics into the active MLflow run context.
- MlflowCheckpointRouter: Redirects checkpoint output into the active MLflow run.
- NumpyWriter: Persists predict-step outputs to NumPy arrays.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from lightning import Callback
from loguru import logger
from pydantic import DirectoryPath, validate_call

from dlkit.infrastructure.io.url_utils import get_url_path
from dlkit.infrastructure.utils.logging_config import get_logger

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer

prediction_logger = get_logger(__name__)


class TransformFittingCallback(Callback):
    """Fits a batch transformer on the training dataloader before training starts.

    Replaces ``StandardLightningWrapper.on_fit_start()`` with a proper
    Lightning Callback, separating the transform-fitting concern from the
    training loop wrapper.

    The callback is a no-op when the transformer is already fitted or does
    not implement ``IFittableBatchTransformer``.

    Args:
        batch_transformer: The batch transformer to fit. Fitting is skipped
            unless it implements ``IFittableBatchTransformer``.

    Example:
        ```python
        callback = TransformFittingCallback(batch_transformer)
        trainer = Trainer(callbacks=[callback])
        ```
    """

    def __init__(self, batch_transformer: Any) -> None:
        """Initialize with the batch transformer to manage.

        Args:
            batch_transformer: Transformer to fit; may or may not be fittable.
        """
        super().__init__()
        self._batch_transformer = batch_transformer

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Fit the batch transformer if it is fittable and not yet fitted.

        Called automatically by Lightning before the first training epoch.

        Args:
            trainer: The Lightning Trainer driving the fit.
            pl_module: The LightningModule being trained (unused).
        """
        from dlkit.engine.adapters.lightning.protocols import IFittableBatchTransformer

        if not isinstance(self._batch_transformer, IFittableBatchTransformer):
            return
        if self._batch_transformer.is_fitted():
            return
        dm = getattr(trainer, "datamodule", None)
        if dm is None or not hasattr(dm, "train_dataloader"):
            return
        loader = dm.train_dataloader()
        logger.debug("Starting transform fitting from training dataloader.")
        self._batch_transformer.fit(loader)
        logger.debug("Finished transform fitting.")


class MLflowEpochLogger(Callback):
    """Callback that logs metrics to MLflow using epoch numbers as the x-axis."""

    def __init__(self, run_context: Any) -> None:
        super().__init__()
        self.run_context = run_context

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_metrics(trainer, stage="train")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_metrics(trainer, stage="val")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_metrics(trainer, stage="test")

    def _log_metrics(self, trainer: Trainer, stage: str) -> None:
        if not self.run_context or getattr(trainer, "sanity_checking", False):
            return

        try:
            metrics = getattr(trainer, "callback_metrics", None) or {}
            prepared = self._collect_stage_metrics(metrics, stage)
            if not prepared:
                return

            step = getattr(trainer, "current_epoch", 0)
            self.run_context.log_metrics(prepared, step=step)
            logger.debug(
                "Logged {} metrics for stage '{}' at step {}",
                len(prepared),
                stage,
                step,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"Failed to log metrics with epoch logger: {exc}")

    def _collect_stage_metrics(self, metrics: dict[str, Any], stage: str) -> dict[str, float]:
        stage_aliases = {
            "train": ("train", "training"),
            "val": ("val", "valid", "validation"),
            "test": ("test", "testing"),
        }
        prefixes = stage_aliases.get(stage, (stage,))
        collected: dict[str, float] = {}

        for key, raw_value in metrics.items():
            normalized_key = key.lower()
            if normalized_key == "lr" and stage != "train":
                continue

            if self._matches_stage_prefix(key, prefixes):
                metric_key = key
            else:
                if normalized_key.startswith("train") and stage != "train":
                    continue
                if normalized_key.startswith("val") and stage != "val":
                    continue
                if normalized_key.startswith("test") and stage != "test":
                    continue
                metric_key = self._format_metric_key(stage, key)

            numeric = self._to_numeric(raw_value)
            if numeric is None:
                logger.debug(f"Skipping non-numeric metric: {key}")
                continue
            collected[metric_key] = numeric

        return collected

    @staticmethod
    def _format_metric_key(stage: str, key: str) -> str:
        if stage.lower() != "test":
            return key
        suffix = " test"
        if key.lower().endswith(suffix):
            return key
        return f"{key}{suffix}"

    @staticmethod
    def _matches_stage_prefix(metric_key: str, prefixes: tuple[str, ...]) -> bool:
        normalized = metric_key.lower()
        for prefix in prefixes:
            prefix_lower = prefix.lower()
            if normalized.startswith(prefix_lower):
                return True
            if normalized.startswith(f"{prefix_lower}/"):
                return True
            if normalized.startswith(f"{prefix_lower}_"):
                return True
            if normalized.startswith(f"{prefix_lower}."):
                return True
        return False

    @staticmethod
    def _to_numeric(value: Any) -> float | None:
        candidate = value
        if hasattr(candidate, "detach"):
            try:
                candidate = candidate.detach()
            except Exception:  # pragma: no cover - defensive branch
                return None
        if hasattr(candidate, "cpu"):
            try:
                candidate = candidate.cpu()
            except Exception:  # pragma: no cover - defensive branch
                return None
        if hasattr(candidate, "item") and callable(candidate.item):
            try:
                candidate = candidate.item()
            except Exception:  # pragma: no cover - defensive branch
                return None
        if isinstance(candidate, (int, float)):
            return float(candidate)
        if isinstance(candidate, str):
            try:
                return float(candidate)
            except TypeError, ValueError:
                return None
        return None


def _resolve_local_artifact_dir() -> Path | None:
    """Return the local artifact directory for the active MLflow run, or None."""
    try:
        import mlflow

        active_run = mlflow.active_run()
        if active_run is None:
            return None

        artifact_uri = mlflow.get_artifact_uri()
        if not artifact_uri or not artifact_uri.startswith("file://"):
            return None

        raw_path = get_url_path(artifact_uri).lstrip("/")
        candidate = Path("/" + raw_path) if not Path(raw_path).is_absolute() else Path(raw_path)
        return candidate.resolve()
    except Exception as exc:
        logger.debug(f"MlflowCheckpointRouter: could not resolve artifact dir: {exc}")
        return None


def _redirect_checkpoint_callbacks(trainer: Trainer, checkpoint_dir: Path) -> None:
    """Set ``dirpath`` on unset ModelCheckpoint callbacks."""
    try:
        from lightning.pytorch.callbacks import ModelCheckpoint
    except ImportError:
        try:
            from pytorch_lightning.callbacks import ModelCheckpoint
        except ImportError:
            logger.debug("MlflowCheckpointRouter: ModelCheckpoint not available")
            return

    for cb in getattr(trainer, "callbacks", []):
        if isinstance(cb, ModelCheckpoint) and cb.dirpath is None:
            cb.dirpath = str(checkpoint_dir)
            logger.debug(f"MlflowCheckpointRouter: redirected ModelCheckpoint -> {checkpoint_dir}")


class MlflowCheckpointRouter(Callback):
    """Redirect unset ModelCheckpoint dirpaths into the MLflow artifact store."""

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        artifact_dir = _resolve_local_artifact_dir()
        if artifact_dir is None:
            logger.debug(
                "MlflowCheckpointRouter: no local MLflow artifact dir found; skipping redirect"
            )
            return

        checkpoint_dir = artifact_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _redirect_checkpoint_callbacks(trainer, checkpoint_dir)


class NumpyWriter(Callback):
    """Accumulate prediction outputs and persist them as NumPy arrays."""

    @validate_call
    def __init__(
        self, output_dir: DirectoryPath | None = None, filenames: Sequence[str] = ("predictions",)
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self._use_mlflow = False

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._predictions: dict[str, list[torch.Tensor]] = {}
        self._filenames: Sequence[str] = filenames

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.output_dir is None:
            self.output_dir, self._use_mlflow = self._resolve_default_output_dir()
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Mapping[str, torch.Tensor],
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if isinstance(outputs, Mapping):
            for i, (key, value) in enumerate(outputs.items()):
                write_key = self._filenames[i] if len(self._filenames) > i else key
                self._store_predictions(write_key, value)
        elif isinstance(outputs, list | tuple):
            for i, value in enumerate(outputs):
                self._store_predictions(self._filenames[i], value)
        elif isinstance(outputs, torch.Tensor):
            self._store_predictions(self._filenames[0], outputs)
        else:
            prediction_logger.error("Unexpected output type in NumpyWriter: {}", type(outputs))

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._predictions:
            prediction_logger.error("No predictions accumulated in NumpyWriter.")
            return

        for key, tensor_list in self._predictions.items():
            concatenated = torch.cat(tensor_list, dim=0).cpu().numpy()
            if self.output_dir is None:
                prediction_logger.error("Output directory is None, cannot save predictions")
                continue
            output_path = Path(self.output_dir) / f"{key}.npy"
            try:
                np.save(output_path, concatenated)
                if self._use_mlflow:
                    import mlflow

                    current_run = mlflow.active_run()
                    if current_run is not None:
                        mlflow.log_artifact(str(output_path), artifact_path="predictions")

                prediction_logger.debug("Saved prediction output {}", output_path)
            except OSError as exc:
                prediction_logger.error("Failed to save prediction output {}: {}", output_path, exc)
                continue

    def _store_predictions(self, key: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            prediction_logger.warning("Output for key '{}' is not a torch.Tensor; skipping", key)
            return
        if key not in self._predictions:
            self._predictions[key] = []
        self._predictions[key].append(value)

    @staticmethod
    def _resolve_default_output_dir() -> tuple[Path, bool]:
        from dlkit.infrastructure.io.locations import predictions_dir

        try:
            import mlflow

            if mlflow.active_run() is None:
                return predictions_dir(), False
            artifact_uri = mlflow.get_artifact_uri()
            if artifact_uri and artifact_uri.startswith("file://"):
                return Path(get_url_path(artifact_uri).lstrip("/")), True
        except Exception:
            pass

        return predictions_dir(), False
