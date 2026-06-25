"""Compatibility shim: maps old workflow config names to existing settings classes.

These backward-compatible names are used by engine code and tests that have not yet
been migrated to the new JobConfig field paths. Will be removed in Task 5.

The UPPERCASE-field classes (SESSION, MODEL, DATASET, TRAINING, MLFLOW, OPTUNA) are
provided via BaseWorkflowSettings and its subclasses. The new JobConfig types
(TrainingJobConfig, SearchJobConfig, InferenceJobConfig) have lowercase fields.
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

# Old names used by engine files. These still accept UPPERCASE fields (SESSION, TRAINING, etc.)
TrainingWorkflowConfig = TrainingWorkflowSettings
InferenceWorkflowConfig = InferenceWorkflowSettings
# Optimization: TrainingWorkflowSettings is the closest backward-compatible type
OptimizationWorkflowConfig = TrainingWorkflowSettings

__all__ = [
    "InferenceWorkflowConfig",
    "JobConfig",
    "OptimizationWorkflowConfig",
    "TrainingWorkflowConfig",
    "TrainingJobConfig",
    "SearchJobConfig",
    "InferenceJobConfig",
]
