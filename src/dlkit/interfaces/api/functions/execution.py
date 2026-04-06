"""Unified execution function with intelligent workflow routing."""

from __future__ import annotations

from typing import Any, cast

from dlkit.common import OptimizationResult, TrainingResult
from dlkit.common.hooks import LifecycleHooks
from dlkit.engine.workflows.entrypoints import execute as runtime_execute
from dlkit.infrastructure.config import GeneralSettings
from dlkit.infrastructure.config.protocols import BaseSettingsProtocol
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from dlkit.interfaces.api.domain.override_types import ExecutionOverrides


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
