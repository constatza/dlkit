"""Shared naming utilities for MLflow tracking following DRY and SRP.

Single source of truth for experiment and run naming logic across all workflows.
"""

from importlib import import_module
from typing import Any
from uuid import uuid4

from dlkit.infrastructure.config import GeneralSettings
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)

type _WorkflowSettings = GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig


def determine_experiment_name(settings: _WorkflowSettings, mlflow_config: Any = None) -> str:
    """Determine experiment name using priority chain with guard clauses.

    Priority: MLFLOW.experiment_name → SESSION.name (with default "dlkit-session")
    (Explicit MLflow config overrides session name)

    Args:
        settings: Configuration settings
        mlflow_config: Optional MLflow configuration

    Returns:
        Experiment name (always returns a value via SESSION.name default)
    """
    if mlflow_config:
        experiment_name = getattr(mlflow_config, "experiment_name", None)
        if isinstance(experiment_name, str):
            normalized = experiment_name.strip()
            if normalized and normalized != "Experiment":
                return normalized

    # Use SESSION.name (which has default "dlkit-session" in SessionSettings)
    # This always returns a value since SESSION.name has a default
    return settings.SESSION.name


def determine_study_name(settings: _WorkflowSettings, optuna_config: Any) -> str:
    """Determine study name using priority chain with guard clauses.

    The study name is used as the MLflow parent run name. Explicit configuration
    wins; otherwise we fall back to MLflow-style random names to avoid mirroring
    the experiment name.

    Args:
        settings: Configuration settings
        optuna_config: Optuna configuration

    Returns:
        Study name for Optuna workflows
    """
    # Guard: Check MLFLOW.run_name for explicit tracking overrides
    mlflow_config = getattr(settings, "MLFLOW", None)
    if mlflow_config:
        run_name = getattr(mlflow_config, "run_name", None)
        if isinstance(run_name, str) and run_name.strip():
            return run_name.strip()

    # Guard: Check OPTUNA.study_name
    study_name = getattr(optuna_config, "study_name", None)
    if study_name:
        return study_name

    return _generate_random_run_name()


def _generate_random_run_name() -> str:
    """Generate MLflow-style random run name with graceful fallback."""
    try:
        name_utils = import_module("mlflow.utils.name_utils")
        generator = getattr(name_utils, "_generate_random_name", None)
        if callable(generator):
            return str(generator())
    except Exception:
        # Last resort: use simple UUID fragment to avoid collisions
        return f"dlkit-run-{uuid4().hex[:8]}"

    # If MLflow name generator missing, fall back to UUID-based slug
    return f"dlkit-run-{uuid4().hex[:8]}"
