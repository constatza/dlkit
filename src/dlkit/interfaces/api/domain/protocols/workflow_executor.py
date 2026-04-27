"""Protocol for workflow execution abstraction."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dlkit.common import OptimizationResult, TrainingResult
from dlkit.common.hooks import LifecycleHooks
from dlkit.engine.workflows.entrypoints._settings import WorkflowSettings
from dlkit.interfaces.api.domain.override_types import (
    ExecutionOverrides,
    OptimizationOverrides,
    TrainingOverrides,
)


@runtime_checkable
class IWorkflowExecutor(Protocol):
    """Protocol for workflow execution engines."""

    def train(
        self,
        settings: WorkflowSettings,
        overrides: TrainingOverrides | None = None,
        *,
        mlflow: bool = False,
        hooks: LifecycleHooks | None = None,
    ) -> TrainingResult:
        """Execute a training workflow.

        Args:
            settings: Workflow configuration settings.
            overrides: Optional training overrides.
            mlflow: If True, ensure MLFLOW section exists in settings.
            hooks: Optional lifecycle hooks for training events.

        Returns:
            TrainingResult containing trained model state and metrics.
        """
        ...

    def optimize(
        self,
        settings: WorkflowSettings,
        overrides: OptimizationOverrides | None = None,
        *,
        mlflow: bool = False,
    ) -> OptimizationResult:
        """Execute a hyperparameter optimization workflow.

        Args:
            settings: Workflow configuration settings.
            overrides: Optional optimization overrides.
            mlflow: If True, ensure MLFLOW section exists in settings.

        Returns:
            OptimizationResult containing best model and trial history.
        """
        ...

    def execute(
        self,
        settings: WorkflowSettings,
        overrides: ExecutionOverrides | None = None,
        *,
        mlflow: bool = False,
        hooks: LifecycleHooks | None = None,
    ) -> TrainingResult | OptimizationResult:
        """Execute a workflow with intelligent routing based on settings.

        Args:
            settings: Workflow configuration settings.
            overrides: Optional execution overrides.
            mlflow: If True, ensure MLFLOW section exists in settings.
            hooks: Optional lifecycle hooks for training events.

        Returns:
            TrainingResult or OptimizationResult depending on workflow type.
        """
        ...
