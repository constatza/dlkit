"""Runtime-owned training workflow entrypoint."""

from __future__ import annotations

from dlkit.common import TrainingResult
from dlkit.common.errors import WorkflowError
from dlkit.common.hooks import LifecycleHooks
from dlkit.engine.workflows.orchestrator import Orchestrator
from dlkit.infrastructure.utils.logging_config import get_logger

from ._entrypoint_context import EntrypointContext
from ._override_types import TrainingOverrides
from ._settings import WorkflowSettings

logger = get_logger(__name__)


def train(
    settings: WorkflowSettings,
    overrides: TrainingOverrides | None = None,
    *,
    hooks: LifecycleHooks | None = None,
) -> TrainingResult:
    """Run a training workflow through runtime orchestration."""
    context = EntrypointContext.prepare(settings, overrides, workflow_name="training")

    try:
        orchestrator = Orchestrator()
        execution_result = context.run_with_path_context(
            lambda: orchestrator.execute_training(context.settings, hooks=hooks)
        )
        duration = context.elapsed()
        return TrainingResult(
            model_state=getattr(execution_result, "model_state", None),
            metrics=getattr(execution_result, "metrics", None) or {},
            artifacts=getattr(execution_result, "artifacts", None) or {},
            duration_seconds=duration
            if duration > 0
            else getattr(execution_result, "duration_seconds", 0.0),
            predictions=getattr(execution_result, "predictions", None),
            mlflow_run_id=getattr(execution_result, "mlflow_run_id", None),
            mlflow_tracking_uri=getattr(execution_result, "mlflow_tracking_uri", None),
        )
    except Exception as exc:
        if isinstance(exc, WorkflowError):
            raise
        logger.error("Training execution failed: {}", exc)
        raise WorkflowError(
            f"Training execution failed: {exc!s}",
            {"workflow": "training", "error": str(exc)},
        ) from exc
