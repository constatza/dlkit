"""Pure vanilla training execution following SRP."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from lightning.pytorch import LightningDataModule, LightningModule, Trainer

from dlkit.common import ModelState, TrainingResult
from dlkit.common.errors import WorkflowError
from dlkit.common.protocols import IDataModule, ITrainableModule
from dlkit.domain.metrics.collect import collect_metrics
from dlkit.engine.training.components import RuntimeComponents
from dlkit.infrastructure.config.core.base_settings import BasicSettings
from dlkit.infrastructure.config.core.updater import update_settings
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from dlkit.infrastructure.precision.service import get_precision_service
from dlkit.infrastructure.utils.logging_config import get_logger

from .interfaces import ITrainingExecutor

logger = get_logger(__name__)


@contextmanager
def _suppress_training_runtime_warnings():
    """Suppress known framework warnings that add noise during successful runs."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*weights_only.*", category=UserWarning)
        warnings.filterwarnings(
            "ignore",
            message="The '.*_dataloader' does not have many workers.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*",
            category=UserWarning,
        )
        yield


class VanillaExecutor(ITrainingExecutor):
    """Pure training execution without any tracking or optimization concerns.

    Single responsibility: Execute PyTorch Lightning training workflow.
    """

    def execute(
        self,
        components: RuntimeComponents,
        settings: TrainingWorkflowConfig | OptimizationWorkflowConfig,
    ) -> TrainingResult:
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

            # Log precision information for debugging
            precision_service = get_precision_service()
            precision_info = precision_service.get_precision_info(settings.SESSION)
            logger.debug(
                "Training precision strategy='{}' torch_dtype='{}'",
                precision_info.get("strategy"),
                precision_info.get("torch_dtype"),
            )

            # Apply automatic learning rate tuning if configured
            self._apply_lr_tuning(trainer, model, datamodule, settings)

            # Determine if we should resume from checkpoint
            ckpt_path = self._get_resume_checkpoint_path(settings)

            # Core training execution
            # Use weights_only=False for dlkit checkpoints which may contain custom classes
            with _suppress_training_runtime_warnings():
                trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

            # Optional post-training steps (best effort)
            predictions = self._run_optional_steps(trainer, model, datamodule)

            # Collect metrics and artifacts
            metrics = self._collect_metrics(trainer)
            artifacts = self._collect_artifacts(trainer)

            return TrainingResult(
                model_state=ModelState(
                    model=cast(ITrainableModule, model),
                    datamodule=cast(IDataModule, datamodule),
                    trainer=trainer,
                    settings=settings,
                ),
                metrics=metrics,
                artifacts=artifacts,
                duration_seconds=0.0,
                predictions=predictions,
            )

        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            raise WorkflowError(
                f"Vanilla execution failed: {e}\n{tb}",
                {"stage": "execute", "trace": tb},
            ) from e

    def _apply_lr_tuning(
        self,
        trainer: Trainer,
        model: LightningModule,
        datamodule: LightningDataModule | None,
        settings: TrainingWorkflowConfig | OptimizationWorkflowConfig,
    ) -> None:
        """Apply automatic learning rate tuning if configured.

        LR tuning is enabled by the presence of TRAINING.lr_tuner settings.
        If the section is absent (None), tuning is skipped.

        Args:
            trainer: PyTorch Lightning trainer
            model: Lightning module
            datamodule: Optional datamodule
            settings: Global training settings
        """
        if not settings.TRAINING:
            return

        lr_tuner_settings = getattr(settings.TRAINING, "lr_tuner", None)
        if lr_tuner_settings is None:
            return

        from dlkit.engine.training.tuning import LRTuner

        lr_tuner = LRTuner()
        try:
            suggested_lr = lr_tuner.tune(trainer, model, lr_tuner_settings, datamodule)

            if self._try_set_model_lr(model, suggested_lr):
                return
            if self._try_set_optimizer_lr(model, suggested_lr):
                return
            logger.warning(
                "Could not update learning rate: model.optimizer.lr not accessible. "
                "Ensure your model uses OptimizerSettings with an 'lr' field."
            )

        except Exception as e:
            logger.warning(
                "Learning rate tuning failed: %s. Continuing with configured learning rate.", e
            )

    def _try_set_model_lr(self, model: LightningModule, lr: float) -> bool:
        """Try to set learning rate via model attribute.

        Attempts to set model.lr property and verifies that the optimizer
        learning rate was updated accordingly.

        Args:
            model: Lightning module
            lr: Learning rate value to set

        Returns:
            True if learning rate was successfully set, False otherwise
        """
        if not hasattr(model, "lr"):
            return False
        try:
            object.__setattr__(model, "lr", lr)
            if getattr(getattr(model, "optimizer", None), "lr", None) == lr:
                logger.info("Learning rate tuned to {}", lr)
                return True
        except Exception:
            pass
        return False

    def _try_set_optimizer_lr(self, model: LightningModule, lr: float) -> bool:
        """Try to set learning rate via optimizer settings.

        Attempts to update model.optimizer (BasicSettings instance) with
        the new learning rate value.

        Args:
            model: Lightning module
            lr: Learning rate value to set

        Returns:
            True if learning rate was successfully set, False otherwise
        """
        if not hasattr(model, "optimizer") or not hasattr(model.optimizer, "lr"):
            return False
        optimizer_attr = model.optimizer
        if not isinstance(optimizer_attr, BasicSettings):
            return False
        updated = update_settings(optimizer_attr, {"lr": lr})
        object.__setattr__(model, "optimizer", updated)
        logger.info("Learning rate tuned to {}", lr)
        return True

    def _run_optional_steps(
        self,
        trainer: Trainer,
        model: LightningModule,
        datamodule: LightningDataModule | None,
    ) -> list[Any] | None:
        """Run optional post-training steps (predict, test); return predictions.

        Args:
            trainer: PyTorch Lightning trainer
            model: Lightning module
            datamodule: Optional datamodule

        Returns:
            Prediction batches from trainer.predict(), or None if predict failed.
        """
        predictions = None
        try:
            with _suppress_training_runtime_warnings():
                predictions = trainer.predict(model, datamodule=datamodule)
        except Exception as e:
            logger.debug("Post-training predict step failed (non-fatal): %s", e)

        try:
            with _suppress_training_runtime_warnings():
                trainer.test(model, datamodule=datamodule)
        except Exception as e:
            logger.debug("Post-training test step failed (non-fatal): %s", e)

        return predictions

    def _collect_metrics(self, trainer: Trainer) -> dict[str, Any]:
        """Collect metrics from trainer after training.

        Args:
            trainer: PyTorch Lightning trainer

        Returns:
            Dictionary of collected metrics
        """
        metrics: dict[str, Any] = {}
        callback_metrics = getattr(trainer, "callback_metrics", None)
        progress_metrics = getattr(trainer, "progress_bar_metrics", None)
        logged_metrics = getattr(trainer, "logged_metrics", None)

        # callback_metrics is the most complete view of end-of-epoch values
        metrics.update(collect_metrics(callback_metrics))
        metrics.update(collect_metrics(progress_metrics))
        metrics.update(collect_metrics(logged_metrics))

        return metrics

    def _collect_artifacts(self, trainer: Trainer) -> dict[str, Path]:
        """Collect checkpoint artifacts from trainer callbacks.

        Args:
            trainer: PyTorch Lightning trainer

        Returns:
            Dictionary mapping artifact names to paths
        """
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
                        last_checkpoints = list(dirpath.glob("last.ckpt")) + list(
                            dirpath.glob("*-last.ckpt")
                        )
                        if last_checkpoints:
                            artifacts["last_checkpoint"] = last_checkpoints[0]
                        # Check for any .ckpt files as fallback for best checkpoint
                        elif not artifacts.get("best_checkpoint"):
                            ckpt_files = [f for f in dirpath.glob("*.ckpt") if "last" not in f.name]
                            if ckpt_files:
                                artifacts["best_checkpoint"] = ckpt_files[0]

        return artifacts

    def _get_resume_checkpoint_path(
        self,
        settings: TrainingWorkflowConfig | OptimizationWorkflowConfig,
    ) -> str | None:
        """Get checkpoint path for resuming training if configured.

        Checks TRAINING.resume_from_checkpoint for the checkpoint path.

        Args:
            settings: Global training settings

        Returns:
            Checkpoint path as string if resuming is configured, None otherwise
        """
        # Check TRAINING.resume_from_checkpoint
        training_checkpoint = (
            getattr(settings.TRAINING, "resume_from_checkpoint", None)
            if hasattr(settings, "TRAINING") and settings.TRAINING
            else None
        )
        if training_checkpoint is None:
            return None

        checkpoint_path = Path(training_checkpoint)
        if not checkpoint_path.exists():
            logger.warning(
                "Training checkpoint configured but not found: %s. Starting training from scratch.",
                checkpoint_path,
            )
            return None

        logger.info("Resuming training from checkpoint: %s", checkpoint_path)
        return str(checkpoint_path)
