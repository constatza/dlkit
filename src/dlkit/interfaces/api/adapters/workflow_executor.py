"""Adapter delegating workflow execution to engine entrypoints."""

from __future__ import annotations

from dlkit.common import OptimizationResult, TrainingResult
from dlkit.common.hooks import LifecycleHooks
from dlkit.engine.workflows.entrypoints import (
    execute as runtime_execute,
)
from dlkit.engine.workflows.entrypoints import (
    optimize as runtime_optimize,
)
from dlkit.engine.workflows.entrypoints import (
    train as runtime_train,
)
from dlkit.engine.workflows.entrypoints._settings import WorkflowSettings
from dlkit.infrastructure.config.job_config import (
    SearchJobConfig,
    TrainingJobConfig,
)
from dlkit.interfaces.api.domain.override_types import (
    ExecutionOverrides,
    OptimizationOverrides,
    TrainingOverrides,
)


def _apply_mlflow_flag(settings: WorkflowSettings, mlflow: bool) -> WorkflowSettings:
    """Return settings with tracking backend ensured when mlflow=True.

    When ``mlflow=True`` and the config has no explicit tracking backend,
    patches the tracking section to use MLflow so the engine enables it.

    Args:
        settings: Workflow configuration settings.
        mlflow: Whether to ensure MLflow tracking is configured.

    Returns:
        Settings with tracking backend set to ``"mlflow"``, or original
        settings if ``mlflow=False`` or tracking is already configured.
    """
    if not mlflow:
        return settings
    tracking = getattr(settings, "tracking", None)
    if tracking is not None and getattr(tracking, "backend", None) not in (None, "none"):
        return settings
    return settings.patch({"tracking": {"backend": "mlflow"}})


class EngineWorkflowExecutor:
    """Concrete executor adapter for engine workflow entrypoints."""

    def train(
        self,
        settings: WorkflowSettings,
        overrides: TrainingOverrides | None = None,
        *,
        mlflow: bool = False,
        hooks: LifecycleHooks | None = None,
    ) -> TrainingResult:
        """Execute a training workflow via engine entrypoint.

        Args:
            settings: Training workflow configuration settings.
            overrides: Optional training overrides.
            mlflow: If True, ensure MLflow tracking is configured.
            hooks: Optional lifecycle hooks for training events.

        Returns:
            TrainingResult containing trained model state and metrics.

        Raises:
            TypeError: If settings is not a TrainingJobConfig.
        """
        settings_with_tracking = _apply_mlflow_flag(settings, mlflow)
        if not isinstance(settings_with_tracking, TrainingJobConfig):
            raise TypeError(
                f"train() requires TrainingJobConfig, got {type(settings_with_tracking).__name__}"
            )
        return runtime_train(settings_with_tracking, overrides=overrides, hooks=hooks)

    def optimize(
        self,
        settings: WorkflowSettings,
        overrides: OptimizationOverrides | None = None,
        *,
        mlflow: bool = False,
    ) -> OptimizationResult:
        """Execute optimization via engine entrypoint.

        Args:
            settings: Optimization workflow configuration settings.
            overrides: Optional optimization overrides.
            mlflow: If True, ensure MLflow tracking is configured.

        Returns:
            OptimizationResult containing best model and trial history.

        Raises:
            TypeError: If settings is not a SearchJobConfig.
        """
        settings_with_tracking = _apply_mlflow_flag(settings, mlflow)
        if not isinstance(settings_with_tracking, SearchJobConfig):
            raise TypeError(
                f"optimize() requires SearchJobConfig, got {type(settings_with_tracking).__name__}"
            )
        return runtime_optimize(settings_with_tracking, overrides=overrides)

    def execute(
        self,
        settings: WorkflowSettings,
        overrides: ExecutionOverrides | None = None,
        *,
        mlflow: bool = False,
        hooks: LifecycleHooks | None = None,
    ) -> TrainingResult | OptimizationResult:
        """Execute a workflow with intelligent routing via engine entrypoint.

        Args:
            settings: Workflow configuration settings.
            overrides: Optional execution overrides.
            mlflow: If True, ensure MLflow tracking is configured.
            hooks: Optional lifecycle hooks for training events.

        Returns:
            TrainingResult or OptimizationResult depending on workflow type.
        """
        settings_with_tracking = _apply_mlflow_flag(settings, mlflow)
        return runtime_execute(settings_with_tracking, overrides=overrides, hooks=hooks)
