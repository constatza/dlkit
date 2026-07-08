"""Lightweight, tracking-adjacent execution helpers for high-volume training loops.

Collapses two previously-duplicated partial reimplementations of "prepare a
trainer for tracked execution" - ``TrialExecutor._disable_checkpoints``/
``_inject_mlflow_logger`` (HPO trial loop) and
``MultiRunOrchestrator._inject_epoch_logger`` (multirun sweeps) - into one
shared module.

These helpers are intentionally a strict subset of what
:class:`~dlkit.engine.tracking.tracking_decorator.TrackingDecorator` provides:
no plots, no checkpoints, no :class:`~dlkit.engine.tracking.artifact_logger
.ArtifactLogger`, no :class:`~dlkit.engine.tracking.settings_logger
.SettingsLogger`, no :class:`~dlkit.engine.tracking.dataset_logger
.DatasetLogger`. That is a deliberate performance tradeoff for high-volume,
repeated training (e.g. hundreds of Optuna trials), not a bug. Any caller
whose artifacts matter - an ordinary ``train()`` call, an HPO best-retrain, or
a multirun child run - must go through ``TrackingDecorator``/
``TrackingDecorator.execute_within_run`` instead of this module.
"""

from __future__ import annotations

from dlkit.common import TrainingResult
from dlkit.common.hooks import LifecycleHooks
from dlkit.engine.tracking.interfaces import IRunContext
from dlkit.engine.training.components import RuntimeComponents
from dlkit.engine.training.vanilla_executor import VanillaExecutor
from dlkit.infrastructure.utils.logging_config import get_logger

logger = get_logger(__name__)


def disable_checkpoint_callbacks(components: RuntimeComponents) -> None:
    """Remove checkpoint callbacks from the trainer.

    During high-volume repeated training (e.g. optimization trials), we don't
    want to save checkpoints for every run as they take up disk space. Only
    the final best model (via the full ``TrackingDecorator`` path) should be
    checkpointed.

    Args:
        components: Build components with trainer.
    """
    from pytorch_lightning.callbacks import ModelCheckpoint

    trainer = getattr(components, "trainer", None)
    if not trainer or not hasattr(trainer, "callbacks"):
        return

    try:
        original_count = len(trainer.callbacks)
        trainer.callbacks = [cb for cb in trainer.callbacks if not isinstance(cb, ModelCheckpoint)]
        removed_count = original_count - len(trainer.callbacks)

        if removed_count > 0:
            logger.debug("Disabled %s checkpoint callback(s) for optimization trial", removed_count)

    except Exception as e:
        logger.warning("Failed to disable checkpoints: {}", e)


def inject_epoch_metric_logger(components: RuntimeComponents, run_context: IRunContext) -> None:
    """Append an ``MLflowEpochLogger`` to the trainer's callbacks.

    This is the lightweight subset of what ``TrackingDecorator
    ._inject_mlflow_callbacks`` does: only the epoch metric logger, no
    plot or checkpoint-dir callbacks.

    Args:
        components: Build components with a mutable ``trainer.callbacks`` list.
        run_context: Already-open run context used as the metric sink.
    """
    try:
        trainer = getattr(components, "trainer", None)
        if not trainer:
            return

        from dlkit.engine.adapters.lightning.callbacks import MLflowEpochLogger

        if not hasattr(trainer, "callbacks"):
            trainer.callbacks = []
        trainer.callbacks.append(MLflowEpochLogger(run_context))
        logger.debug(
            "Injected MLflow epoch logger for run '{}'",
            getattr(run_context, "run_id", "unknown"),
        )

    except Exception as e:
        logger.warning("Failed to inject MLflow epoch logger: {}", e)


def execute_lightweight(
    components: RuntimeComponents,
    settings: object,
    run_context: IRunContext | None,
) -> TrainingResult:
    """Execute training with lightweight, high-volume-friendly tracking.

    This is an intentional performance tradeoff for high-volume repeated
    training (e.g. hundreds of Optuna trials): it disables checkpoint
    callbacks and, when a run context is available, injects only the epoch
    metric logger before delegating to a bare ``VanillaExecutor``. It skips
    plots, checkpoints, ``ArtifactLogger``, ``SettingsLogger``, and
    ``DatasetLogger`` entirely.

    This is NOT a bug and NOT a substitute for full tracking parity. Callers
    whose artifacts matter - an ordinary ``train()`` call, an HPO
    best-retrain, or a multirun child run - must use ``TrackingDecorator``/
    ``TrackingDecorator.execute_within_run`` instead of this function.

    Args:
        components: Pre-built training components.
        settings: Global training settings (must be a JobConfig instance).
        run_context: Already-open run context for epoch metric logging, or
            ``None`` if this run is untracked.

    Returns:
        TrainingResult from the underlying ``VanillaExecutor``.
    """
    disable_checkpoint_callbacks(components)
    if run_context is not None:
        inject_epoch_metric_logger(components, run_context)
    return VanillaExecutor().execute(components, settings)


def fire_post_training_hooks(
    hooks: LifecycleHooks | None,
    run_context: IRunContext | None,
    result: TrainingResult,
) -> None:
    """Fire the post-training subset of ``LifecycleHooks`` on the lightweight path.

    Mirrors what ``TrackingDecorator._run_within_context`` already does for
    the same 4 hooks, so the same hook-firing behavior is available on the
    lightweight path where ``TrackingDecorator`` isn't used at all.

    Args:
        hooks: Optional lifecycle hooks; no-ops entirely when ``None``.
        run_context: Already-open run context to log params/tags/artifacts
            against; no-ops entirely when ``None`` (untracked run).
        result: Training result passed to each hook callable.
    """
    if hooks is None or run_context is None:
        return
    if hooks.on_training_complete:
        hooks.on_training_complete(result)
    if hooks.extra_params:
        run_context.log_params(hooks.extra_params(result))
    if hooks.extra_tags:
        for key, value in hooks.extra_tags(result).items():
            run_context.set_tag(key, value)
    if hooks.extra_artifacts:
        for path in hooks.extra_artifacts(result):
            run_context.log_artifact(path)
