"""Shared helpers for runtime workflow entrypoints."""

from __future__ import annotations

from dlkit.infrastructure.config.job_config import (
    InferenceJobConfig,
    JobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)

# Type alias: union of all concrete job config types accepted by entrypoints
type WorkflowSettings = TrainingJobConfig | SearchJobConfig | InferenceJobConfig | JobConfig
