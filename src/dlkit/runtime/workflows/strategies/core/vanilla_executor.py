"""Pure vanilla training execution following SRP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from loguru import logger

from dlkit.interfaces.api.domain import ModelState, TrainingResult, WorkflowError
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.core.updater import update_settings
from dlkit.tools.utils.metrics import collect_metrics
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

            # Log precision information for debugging
            precision_service = get_precision_service()
            precision_info = precision_service.get_precision_info(settings.SESSION)
            logger.info(f"Training with precision configuration: {precision_info}")

            # Apply automatic learning rate tuning if configured
            self._apply_lr_tuning(trainer, model, datamodule, settings)

            # Determine if we should resume from checkpoint
            ckpt_path = self._get_resume_checkpoint_path(settings)

            # Core training execution
            # Use weights_only=False for dlkit checkpoints which may contain custom classes
            trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

            # Optional post-training steps (best effort)
            predictions = self._run_optional_steps(trainer, model, datamodule)

            # Collect metrics and artifacts
            metrics = self._collect_metrics(trainer)
            artifacts = self._collect_artifacts(trainer)

            return TrainingResult(
                model_state=ModelState(
                    model=model,
                    datamodule=datamodule,
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
        settings: GeneralSettings,
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

        from dlkit.runtime.workflows.strategies.tuning import LRTuner

        lr_tuner = LRTuner()
        try:
            suggested_lr = lr_tuner.tune(trainer, model, lr_tuner_settings, datamodule)

            lr_handled = False

            # Try model-level attribute updates first (Lightning wrappers expose properties)
            if hasattr(model, "lr"):
                try:
                    model.lr = suggested_lr
                    optimizer_lr = getattr(getattr(model, "optimizer", None), "lr", None)
                    if optimizer_lr == suggested_lr:
                        lr_handled = True
                        logger.info(f"Updated model learning rate to {suggested_lr}")
                except Exception:
                    lr_handled = False

            # Fallback to optimizer settings update when attribute path not available or ineffective
            if not lr_handled and hasattr(model, "optimizer") and hasattr(model.optimizer, "lr"):
                model.optimizer = update_settings(model.optimizer, {"lr": suggested_lr})
                logger.info(f"Updated optimizer learning rate to {suggested_lr}")
                lr_handled = True

            if not lr_handled:
                logger.warning(
                    "Could not update learning rate: model.optimizer.lr not accessible. "
                    "Ensure your model uses OptimizerSettings with an 'lr' field."
                )

        except Exception as e:
            logger.warning(
                f"Learning rate tuning failed: {e}. Continuing with configured learning rate."
            )

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
            predictions = trainer.predict(model, datamodule=datamodule)
        except Exception as e:
            logger.debug(f"Post-training predict step failed (non-fatal): {e}")

        try:
            trainer.test(model, datamodule=datamodule)
        except Exception as e:
            logger.debug(f"Post-training test step failed (non-fatal): {e}")

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
                            ckpt_files = [
                                f for f in dirpath.glob("*.ckpt") if "last" not in f.name
                            ]
                            if ckpt_files:
                                artifacts["best_checkpoint"] = ckpt_files[0]

        return artifacts

    def _get_resume_checkpoint_path(self, settings: GeneralSettings) -> str | None:
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
                f"Training checkpoint configured but not found: {checkpoint_path}. "
                f"Starting training from scratch."
            )
            return None

        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        return str(checkpoint_path)
