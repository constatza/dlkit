"""Unified execution function with intelligent workflow routing."""

from __future__ import annotations

from typing import Any, cast

from dlkit.interfaces.api.domain.override_types import ExecutionOverrides
from dlkit.runtime.workflows.entrypoints import execute as runtime_execute
from dlkit.shared import OptimizationResult, TrainingResult
from dlkit.shared.hooks import LifecycleHooks
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.protocols import BaseSettingsProtocol
from dlkit.tools.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)


def execute(
    settings: (
        TrainingWorkflowConfig | OptimizationWorkflowConfig | GeneralSettings | BaseSettingsProtocol
    ),
    overrides: ExecutionOverrides | None = None,
    *,
    hooks: LifecycleHooks | None = None,
) -> TrainingResult | OptimizationResult:
    """Execute DLKit workflow with intelligent routing based on settings."""
    return runtime_execute(
        settings=cast(GeneralSettings, settings),
        overrides=cast(Any, overrides),
        hooks=hooks,
    )
