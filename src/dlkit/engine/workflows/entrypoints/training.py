"""Runtime-owned training workflow entrypoint."""

from __future__ import annotations

from typing import cast

from dlkit.common import TrainingResult
from dlkit.common.errors import WorkflowError
from dlkit.common.hooks import LifecycleHooks
from dlkit.engine.workflows.orchestrator import Orchestrator
from dlkit.infrastructure.config.job_config import TrainingJobConfig
from dlkit.infrastructure.utils.error_handling import raise_error
from dlkit.infrastructure.utils.logging_config import get_logger

from ._entrypoint_context import EntrypointContext
from ._override_types import TrainingOverrides, require_override_model

logger = get_logger(__name__)


def train(
    settings: TrainingJobConfig,
    overrides: TrainingOverrides | None = None,
    *,
    hooks: LifecycleHooks | None = None,
) -> TrainingResult:
    """Run a training workflow through runtime orchestration."""
    logger.info(
        "Training | experiment={} run={} model={} max_epochs={}",
        getattr(getattr(settings, "experiment", None), "name", None) or "dlkit-experiment",
        getattr(getattr(settings, "experiment", None), "run_name", None) or "<auto>",
        getattr(getattr(settings, "model", None), "name", None) or "<unknown>",
        getattr(getattr(settings, "training", None), "max_epochs", "?"),
    )
    validated_overrides = require_override_model(overrides, TrainingOverrides)
    context = EntrypointContext.prepare(settings, validated_overrides, workflow_name="training")

    try:
        orchestrator = Orchestrator()
        training_settings = cast(TrainingJobConfig, context.settings)
        execution_result = context.run_with_path_context(
            lambda: orchestrator.execute_training(training_settings, hooks=hooks)
        )
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
    except Exception as exc:
        if isinstance(exc, WorkflowError):
            raise
        raise_error("Training execution failed", exc)
