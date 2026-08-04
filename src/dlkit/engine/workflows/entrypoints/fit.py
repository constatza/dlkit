"""Runtime-owned one-shot fit workflow entrypoint."""

from __future__ import annotations

from typing import cast

from dlkit.common import TrainingResult
from dlkit.common.hooks import LifecycleHooks
from dlkit.engine.workflows.orchestrator import Orchestrator
from dlkit.infrastructure.config.job_config import FitJobConfig
from dlkit.infrastructure.io.path_context import path_override_context
from dlkit.infrastructure.utils.logging_config import get_logger

from ._entrypoint_context import EntrypointContext
from ._override_types import FitOverrides, require_override_model

logger = get_logger(__name__)


def fit(
    settings: FitJobConfig,
    overrides: FitOverrides | None = None,
    *,
    hooks: LifecycleHooks | None = None,
) -> TrainingResult:
    """Run a one-shot, non-gradient fit workflow through runtime orchestration."""
    logger.info(
        "Fitting | experiment={} run={} model={}",
        getattr(getattr(settings, "experiment", None), "name", None) or "dlkit-experiment",
        getattr(getattr(settings, "experiment", None), "run_name", None) or "<auto>",
        getattr(getattr(settings, "model", None), "name", None) or "<unknown>",
    )
    validated_overrides = require_override_model(overrides, FitOverrides)
    context = EntrypointContext.prepare(settings, validated_overrides, workflow_name="fit")

    def run_fit() -> TrainingResult:
        orchestrator = Orchestrator()
        fit_settings = cast(FitJobConfig, context.settings)
        return orchestrator.execute_training(fit_settings, hooks=hooks)

    checkpoints_dir = validated_overrides.checkpoints_dir if validated_overrides else None
    if checkpoints_dir is not None:
        with path_override_context({"checkpoints_dir": checkpoints_dir}):
            execution_result = context.run(run_fit, error_message="Fit execution failed")
    else:
        execution_result = context.run(run_fit, error_message="Fit execution failed")

    duration = context.elapsed()
    if duration <= 0:
        duration = getattr(execution_result, "duration_seconds", 0.0)
    return TrainingResult(
        model_state=getattr(execution_result, "model_state", None),
        metrics=getattr(execution_result, "metrics", None) or {},
        artifacts=getattr(execution_result, "artifacts", None) or {},
        duration_seconds=duration,
        predictions=getattr(execution_result, "predictions", None),
        mlflow_run_id=getattr(execution_result, "mlflow_run_id", None),
        mlflow_tracking_uri=getattr(execution_result, "mlflow_tracking_uri", None),
    )
