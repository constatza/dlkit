"""Pure vanilla training execution following SRP."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch import LightningDataModule, LightningModule, Trainer

from dlkit.common import ModelState, TrainingResult
from dlkit.common.errors import WorkflowError
from dlkit.common.protocols import IDataModule, ITrainableModule
from dlkit.domain.metrics.collect import collect_metrics
from dlkit.engine.training._checkpoint_helpers import _resolve_checkpoint_path
from dlkit.engine.training._executor_helpers import (
    _checkpoint_artifacts_from_callback,
    _get_lr_tuner,
    _get_optimizer,
    _get_seed,
    _get_trainer_settings,
    _reload_best_checkpoint_weights,
    _suppress_training_runtime_warnings,
)
from dlkit.engine.training.checkpoint_utils import find_checkpoint_callback
from dlkit.engine.training.components import RuntimeComponents
from dlkit.engine.training.tuning import ILRTunable, SupportedLRTuningPlan
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.precision.service import get_precision_service
from dlkit.infrastructure.utils.logging_config import get_logger

from .interfaces import ITrainingExecutor

logger = get_logger(__name__)


class VanillaExecutor(ITrainingExecutor):
    """Pure training execution without any tracking or optimization concerns.

    Single responsibility: Execute PyTorch Lightning training workflow.
    """

    def execute(
        self,
        components: RuntimeComponents,
        settings: object,
    ) -> TrainingResult:
        """Execute pure training workflow.

        Args:
            components: Pre-built training components.
            settings: Global training settings (must be a JobConfig instance).

        Returns:
            TrainingResult with metrics and artifacts.

        Raises:
            WorkflowError: If trainer is missing, checkpoint-path resolution fails,
                or ``trainer.fit(...)`` raises.
            TypeError: If settings is not a JobConfig instance.
        """
        if not isinstance(settings, JobConfig):
            raise TypeError(
                f"VanillaExecutor.execute requires a JobConfig, got {type(settings).__name__}"
            )
        # Set reproducible seed from settings
        from lightning.pytorch import seed_everything

        seed_everything(_get_seed(settings), workers=True)

        trainer = components.trainer
        model = components.model
        datamodule = components.datamodule

        if trainer is None:
            raise WorkflowError("Trainer is required for training", {"stage": "execute"})

        # Log precision information for debugging
        precision_service = get_precision_service()
        precision_info = precision_service.get_precision_info(None)
        logger.debug(
            "Training precision strategy='{}' torch_dtype='{}'",
            precision_info.get("strategy"),
            precision_info.get("torch_dtype"),
        )

        # Apply automatic learning rate tuning if configured
        self._apply_lr_tuning(model, datamodule, settings)

        # Determine if we should resume from checkpoint
        ckpt_path = _resolve_checkpoint_path(settings)

        # Core training execution: trainer.fit() and the immediately-following
        # best-checkpoint reload are the same class of Lightning-internal failure
        # surface (corrupt checkpoint file, IO errors, OOM) that callers need
        # translated into a domain-specific WorkflowError.
        try:
            with _suppress_training_runtime_warnings():
                trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

            fit_metrics = self._collect_metrics(trainer)

            # Reload best-checkpoint weights (no-op if no ModelCheckpoint configured)
            _reload_best_checkpoint_weights(model, trainer, datamodule)
        except Exception as e:
            raise WorkflowError(
                f"Vanilla execution failed: {e}",
                {"stage": "fit"},
            ) from e

        # Optional post-training steps (best effort)
        predictions = self._run_optional_steps(trainer, model, datamodule)

        # Collect metrics and artifacts
        metrics = {**fit_metrics, **self._collect_metrics(trainer)}
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

    def _apply_lr_tuning(
        self,
        model: LightningModule,
        datamodule: LightningDataModule | None,
        settings: Any,
    ) -> None:
        """Apply automatic learning rate tuning if configured.

        Args:
            model: Lightning module.
            datamodule: Optional datamodule.
            settings: Global training settings.
        """
        lr_tuner_settings = _get_lr_tuner(settings)
        if lr_tuner_settings is None:
            return

        from dlkit.engine.training.tuning import UnsupportedLRTuningPlan, get_lr_tuning_plan

        optimizer = _get_optimizer(settings)
        if optimizer is None:
            logger.warning("No optimizer configured; skipping LR tuner.")
            return
        tuning_plan = get_lr_tuning_plan(optimizer)
        if isinstance(tuning_plan, UnsupportedLRTuningPlan):
            logger.warning("{} Skipping LR tuner.", tuning_plan.reason)
            return

        if not isinstance(model, ILRTunable):
            logger.warning(
                "Model {} does not implement ILRTunable; LR tuner result cannot be applied.",
                type(model).__name__,
            )
            return

        try:
            suggested_lr = self._find_lr_with_projected_policy(
                model, datamodule, settings, tuning_plan, lr_tuner_settings
            )
        except Exception as e:
            logger.warning(
                "Learning rate tuning failed: {}. Continuing with configured learning rate.", e
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
        settings: Any,
        tuning_plan: SupportedLRTuningPlan,
        lr_tuner_settings: Any,
    ) -> float:
        """Run Lightning LR finder with a projected single-stage optimizer.

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

        trainer_settings = _get_trainer_settings(settings)
        if trainer_settings is None:
            raise WorkflowError(
                "Trainer settings are required for LR tuning", {"stage": "lr_tuning"}
            )

        # overfit_batches/fast_dev_run silently degrade or no-op Lightning's LR range
        # test without raising (limit_train_batches is never reset by lr_find, and
        # fast_dev_run is a hard no-op) — neutralize both for this throwaway trainer only.
        tuning_trainer = trainer_settings.patch(
            {"overfit_batches": 0, "fast_dev_run": False}
        ).build(session=None)

        tuning_controller = build_optimization_controller(
            cast(Any, model).model, tuning_plan.projected_policy
        )

        tunable = cast(ILRTunable, model)
        with tunable.temporarily_use_controller(tuning_controller):
            return LRTuner().tune(tuning_trainer, model, lr_tuner_settings, datamodule)

    def _run_optional_steps(
        self,
        trainer: Trainer,
        model: LightningModule,
        datamodule: LightningDataModule | None,
    ) -> list[Any] | None:
        """Run optional post-training steps (predict, test); return predictions.

        Args:
            trainer: PyTorch Lightning trainer.
            model: Lightning module.
            datamodule: Optional datamodule.

        Returns:
            Prediction batches from trainer.predict(), or None if predict failed.
        """

        def _predict() -> Any:
            with _suppress_training_runtime_warnings():
                return trainer.predict(model, datamodule=datamodule)

        def _test() -> Any:
            with _suppress_training_runtime_warnings():
                return trainer.test(model, datamodule=datamodule)

        predictions = self._try_optional_step("predict", _predict)
        self._try_optional_step("test", _test)

        return predictions

    def _try_optional_step(self, name: str, step: Callable[[], Any]) -> Any | None:
        """Run an optional post-training step, tolerating non-configuration.

        Args:
            name: Human-readable step name used in log messages.
            step: Zero-argument callable performing the optional step.

        Returns:
            The step's return value, or None if it failed.
        """
        try:
            return step()
        except MisconfigurationException as e:
            logger.debug("Post-training {} step not configured, skipping: {}", name, e)
        except Exception as e:
            logger.warning("Post-training {} step failed (non-fatal): {}", name, e)
        return None

    def _collect_metrics(self, trainer: Trainer) -> dict[str, Any]:
        """Collect metrics from trainer after training.

        Args:
            trainer: PyTorch Lightning trainer.

        Returns:
            Dictionary of collected metrics.
        """
        metrics: dict[str, Any] = {}
        callback_metrics = getattr(trainer, "callback_metrics", None)
        progress_metrics = getattr(trainer, "progress_bar_metrics", None)
        logged_metrics = getattr(trainer, "logged_metrics", None)

        metrics.update(collect_metrics(callback_metrics))
        metrics.update(collect_metrics(progress_metrics))
        metrics.update(collect_metrics(logged_metrics))

        return metrics

    def _collect_artifacts(self, trainer: Trainer) -> dict[str, Path]:
        """Collect checkpoint artifacts from trainer callbacks.

        Args:
            trainer: PyTorch Lightning trainer.

        Returns:
            Dictionary mapping artifact names to paths.
        """
        checkpoint_callback = find_checkpoint_callback(trainer)
        if checkpoint_callback is None:
            return {}
        return _checkpoint_artifacts_from_callback(checkpoint_callback)
