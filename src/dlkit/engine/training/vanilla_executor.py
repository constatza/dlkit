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
from dlkit.engine.training.tuning import ILRTunable, SupportedLRTuningPlan
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
            self._apply_lr_tuning(model, datamodule, settings)

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
        model: LightningModule,
        datamodule: LightningDataModule | None,
        settings: TrainingWorkflowConfig | OptimizationWorkflowConfig,
    ) -> None:
        """Apply automatic learning rate tuning if configured.

        Args:
            model: Lightning module.
            datamodule: Optional datamodule.
            settings: Global training settings.
        """
        if not settings.TRAINING:
            return
        lr_tuner_settings = getattr(settings.TRAINING, "lr_tuner", None)
        if lr_tuner_settings is None:
            return

        from dlkit.engine.training.tuning import UnsupportedLRTuningPlan, get_lr_tuning_plan

        tuning_plan = get_lr_tuning_plan(settings.TRAINING.optimizer)
        if isinstance(tuning_plan, UnsupportedLRTuningPlan):
            logger.warning("%s Skipping LR tuner.", tuning_plan.reason)
            return

        if not isinstance(model, ILRTunable):
            logger.warning(
                "Model %s does not implement ILRTunable; LR tuner result cannot be applied.",
                type(model).__name__,
            )
            return

        try:
            suggested_lr = self._find_lr_with_projected_policy(
                model, datamodule, settings, tuning_plan, lr_tuner_settings
            )
        except Exception as e:
            logger.warning(
                "Learning rate tuning failed: %s. Continuing with configured learning rate.", e
            )
            return

        tuning_plan.apply_suggested_lr(cast(ILRTunable, model), suggested_lr)

        actual_lr = getattr(model, "lr", None)
        if actual_lr != suggested_lr:
            logger.warning(
                "LR tuner suggested %.6f but model.lr=%.6f after apply; "
                "the model lr setter may be a no-op.",
                suggested_lr,
                actual_lr,
            )

    def _find_lr_with_projected_policy(
        self,
        model: LightningModule,
        datamodule: LightningDataModule | None,
        settings: TrainingWorkflowConfig | OptimizationWorkflowConfig,
        tuning_plan: SupportedLRTuningPlan,
        lr_tuner_settings: Any,
    ) -> float:
        """Run Lightning LR finder with a projected single-stage optimizer.

        Temporarily swaps ``model._optimization_controller`` with one built from
        ``tuning_plan.projected_policy`` so Lightning's LR finder sees exactly one
        optimizer. The original controller and ``automatic_optimization`` flag are
        always restored in the ``finally`` block.

        Args:
            model: The training model wrapper.
            datamodule: Optional Lightning datamodule.
            settings: Workflow settings used to build a dedicated tuning trainer.
            tuning_plan: Carries the single-stage projected optimizer policy.
            lr_tuner_settings: LR finder hyper-parameters.

        Returns:
            Suggested learning rate from Lightning's LR finder.
        """
        from dlkit.engine.training.optimization.controllers import build_optimization_controller
        from dlkit.engine.training.tuning import LRTuner

        tuning_trainer = settings.TRAINING.trainer.build(settings.SESSION)
        tuning_controller = build_optimization_controller(
            cast(Any, model).model, tuning_plan.projected_policy
        )
        original_controller = cast(Any, model)._optimization_controller
        original_auto_opt = model.automatic_optimization

        try:
            cast(Any, model)._optimization_controller = tuning_controller
            model.automatic_optimization = not tuning_controller.requires_manual_optimization
            return LRTuner().tune(tuning_trainer, model, lr_tuner_settings, datamodule)
        finally:
            cast(Any, model)._optimization_controller = original_controller
            model.automatic_optimization = original_auto_opt

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
