"""Pure vanilla training execution following SRP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.interfaces.api.domain import TrainingResult, WorkflowError
from dlkit.tools.config import GeneralSettings
from dlkit.runtime.workflows.factories.build_factory import BuildComponents
from dlkit.interfaces.api.services.precision_service import get_precision_service

from .interfaces import ITrainingExecutor


class VanillaExecutor(ITrainingExecutor):
    """Pure training execution without any tracking or optimization concerns.

    Single responsibility: Execute PyTorch Lightning training workflow.
    """

    def execute(self, components: BuildComponents, settings: GeneralSettings) -> TrainingResult:
        """Execute pure training workflow.

        Args:
            components: Pre-built training components
            settings: Global training settings

        Returns:
            TrainingResult with metrics and artifacts

        Raises:
            WorkflowError: If training execution fails
        """
        try:
            # Set reproducible seed from settings
            from pytorch_lightning import seed_everything

            seed_everything(settings.SESSION.seed, workers=True)

            trainer = components.trainer
            model = components.model
            datamodule = components.datamodule

            if trainer is None:
                raise WorkflowError("Trainer is required for training", {"stage": "execute"})

            # Ensure model precision is applied consistently
            if model is not None and hasattr(model, "ensure_precision_applied"):
                model.ensure_precision_applied()

            # Log precision information for debugging
            precision_service = get_precision_service()
            precision_info = precision_service.get_precision_info(settings.SESSION)
            from loguru import logger

            logger.info(f"Training with precision configuration: {precision_info}")

            # Core training execution
            trainer.fit(model, datamodule=datamodule)

            # Optional post-training steps (best effort)
            try:
                trainer.predict(model, datamodule=datamodule)
            except Exception:
                pass

            try:
                trainer.test(model, datamodule=datamodule)
            except Exception:
                pass

            def _collect_metrics(source: dict[str, Any] | None) -> dict[str, Any]:
                """Convert a mapping of metric values to plain python types."""
                collected: dict[str, Any] = {}
                if not source:
                    return collected
                for key, value in source.items():
                    try:
                        collected[key] = float(value)
                    except Exception:
                        collected[key] = value
                return collected

            # Extract metrics from trainer
            metrics: dict[str, Any] = {}
            callback_metrics = getattr(trainer, "callback_metrics", None)
            progress_metrics = getattr(trainer, "progress_bar_metrics", None)
            logged_metrics = getattr(trainer, "logged_metrics", None)

            # callback_metrics is the most complete view of end-of-epoch values, so prefer it
            metrics.update(_collect_metrics(callback_metrics))
            metrics.update(_collect_metrics(progress_metrics))
            metrics.update(_collect_metrics(logged_metrics))

            # Collect checkpoint artifacts from ModelCheckpoint callbacks
            artifacts: dict[str, Path] = {}
            from lightning.pytorch.callbacks import ModelCheckpoint

            callbacks = getattr(trainer, "callbacks", None) or []
            for callback in callbacks:
                if isinstance(callback, ModelCheckpoint):
                    # Collect best and last checkpoints from callback
                    if hasattr(callback, "best_model_path") and callback.best_model_path:
                        artifacts["best_checkpoint"] = Path(callback.best_model_path)
                    if hasattr(callback, "last_model_path") and callback.last_model_path:
                        artifacts["last_checkpoint"] = Path(callback.last_model_path)

                    # Fallback: check filesystem directly if paths aren't set
                    if callback.dirpath and not artifacts.get("last_checkpoint"):
                        dirpath = Path(callback.dirpath)
                        if dirpath.exists():
                            # Check for last.ckpt or *-last.ckpt pattern
                            last_checkpoints = list(dirpath.glob("last.ckpt")) + list(dirpath.glob("*-last.ckpt"))
                            if last_checkpoints:
                                artifacts["last_checkpoint"] = last_checkpoints[0]
                            # Check for any .ckpt files as fallback for best checkpoint
                            elif not artifacts.get("best_checkpoint"):
                                ckpt_files = [f for f in dirpath.glob("*.ckpt") if "last" not in f.name]
                                if ckpt_files:
                                    artifacts["best_checkpoint"] = ckpt_files[0]

            return TrainingResult(
                model_state=None,
                metrics=metrics,
                artifacts=artifacts,
                duration_seconds=0.0,
            )

        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            raise WorkflowError(
                f"Vanilla execution failed: {e}\n{tb}",
                {"stage": "execute", "trace": tb},
            ) from e
