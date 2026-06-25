"""Compatibility shim: maps old GeneralSettings to BaseWorkflowSettings.

Tests and engine code still use GeneralSettings with UPPERCASE fields (SESSION, MODEL,
DATASET, TRAINING, MLFLOW, OPTUNA). BaseWorkflowSettings provides that API.
Will be removed in Task 5 when all callers are updated.
"""

from dlkit.infrastructure.config.workflow_settings_base import (
    BaseWorkflowSettings as GeneralSettings,
)

__all__ = ["GeneralSettings"]
