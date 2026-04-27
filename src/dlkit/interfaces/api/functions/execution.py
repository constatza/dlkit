"""Unified execution function with intelligent workflow routing."""

from __future__ import annotations

from dlkit.common import OptimizationResult, TrainingResult
from dlkit.common.hooks import LifecycleHooks
from dlkit.engine.workflows.entrypoints._settings import WorkflowSettings
from dlkit.interfaces.api.adapters import EngineWorkflowExecutor
from dlkit.interfaces.api.domain.override_types import ExecutionOverrides

_executor: EngineWorkflowExecutor = EngineWorkflowExecutor()


def execute(
    settings: WorkflowSettings,
    overrides: ExecutionOverrides | None = None,
    *,
    hooks: LifecycleHooks | None = None,
) -> TrainingResult | OptimizationResult:
    """Execute DLKit workflow with intelligent routing based on settings."""
    return _executor.execute(
        settings=settings,
        overrides=overrides,
        hooks=hooks,
    )
