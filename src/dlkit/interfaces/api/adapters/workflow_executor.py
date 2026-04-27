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
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from dlkit.interfaces.api.domain.override_types import (
    ExecutionOverrides,
    OptimizationOverrides,
    TrainingOverrides,
)


def _apply_mlflow_flag(settings: WorkflowSettings, mlflow: bool) -> WorkflowSettings:
    """Return settings with MLFLOW section ensured when mlflow=True.

    Args:
        settings: Workflow configuration settings.
        mlflow: Whether to ensure MLFLOW config exists.

    Returns:
        Settings with MLFLOW section ensured, or original settings if mlflow=False.
    """
    if mlflow and not getattr(settings, "MLFLOW", None):
        return settings.patch({"MLFLOW": {}})
    return settings


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
            mlflow: If True, ensure MLFLOW section exists in settings.
            hooks: Optional lifecycle hooks for training events.

        Returns:
            TrainingResult containing trained model state and metrics.
        """
        settings_with_mlflow = _apply_mlflow_flag(settings, mlflow)
        if isinstance(settings_with_mlflow, OptimizationWorkflowConfig):
            raise TypeError(
                "train() requires training or general settings, not OptimizationWorkflowConfig"
            )
        return runtime_train(settings_with_mlflow, overrides=overrides, hooks=hooks)

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
            mlflow: If True, ensure MLFLOW section exists in settings.

        Returns:
            OptimizationResult containing best model and trial history.
        """
        settings_with_mlflow = _apply_mlflow_flag(settings, mlflow)
        if isinstance(settings_with_mlflow, TrainingWorkflowConfig):
            raise TypeError(
                "optimize() requires optimization or general settings, not TrainingWorkflowConfig"
            )
        return runtime_optimize(settings_with_mlflow, overrides=overrides)

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
            mlflow: If True, ensure MLFLOW section exists in settings.
            hooks: Optional lifecycle hooks for training events.

        Returns:
            TrainingResult or OptimizationResult depending on workflow type.
        """
        settings_with_mlflow = _apply_mlflow_flag(settings, mlflow)
        return runtime_execute(settings_with_mlflow, overrides=overrides, hooks=hooks)
