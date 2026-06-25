"""Compatibility shim: WorkflowConfig type alias for old config union types.

Will be removed in Task 5 when all callers are updated.
"""

from dlkit.infrastructure.config.job_config import (
    InferenceJobConfig,
    JobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)
from dlkit.infrastructure.config.workflow_settings import (
    InferenceWorkflowSettings,
    TrainingWorkflowSettings,
)

# Union of old-style BaseWorkflowSettings subclasses and new-style JobConfig subtypes
WorkflowConfig = (
    TrainingJobConfig
    | SearchJobConfig
    | InferenceJobConfig
    | JobConfig
    | TrainingWorkflowSettings
    | InferenceWorkflowSettings
)

__all__ = ["WorkflowConfig"]
