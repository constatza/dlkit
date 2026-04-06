"""Shared helpers for runtime workflow entrypoints."""

from __future__ import annotations

from dlkit.infrastructure.config import GeneralSettings
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)

type WorkflowSettings = GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig


def coerce_general_settings(settings: WorkflowSettings) -> GeneralSettings:
    """Coerce workflow settings objects onto the canonical GeneralSettings model."""
    if isinstance(settings, GeneralSettings):
        return settings
    return GeneralSettings.model_validate(settings.model_dump())
