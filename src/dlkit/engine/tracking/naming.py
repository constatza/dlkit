"""Shared naming utilities for MLflow tracking following DRY and SRP.

Single source of truth for experiment and run naming logic across all workflows.
"""

from importlib import import_module
from typing import Any
from uuid import uuid4

from dlkit.infrastructure.config.job_config import JobConfig


def _session_name(settings: JobConfig) -> str:
    """Extract session/experiment name from JobConfig.

    Args:
        settings: A JobConfig instance.

    Returns:
        Experiment name string (always non-empty).
    """
    return settings.experiment.name if settings.experiment else "dlkit-experiment"


def determine_experiment_name(settings: JobConfig, mlflow_config: Any = None) -> str:
    """Determine experiment name using priority chain with guard clauses.

    Priority: mlflow_config.experiment_name → job.experiment.name (with default).
    Explicit MLflow config overrides session name.

    Args:
        settings: A JobConfig instance.
        mlflow_config: Optional MLflow configuration with experiment_name field.

    Returns:
        Experiment name (always returns a non-empty value).
    """
    if mlflow_config is not None:
        experiment_name = getattr(mlflow_config, "experiment_name", None)
        if isinstance(experiment_name, str):
            normalized = experiment_name.strip()
            if normalized and normalized != "Experiment":
                return normalized

    return _session_name(settings)


def determine_study_name(settings: JobConfig, optuna_config: Any) -> str:
    """Determine study name using priority chain with guard clauses.

    The study name is used as the MLflow parent run name. Explicit configuration
    wins; otherwise we fall back to MLflow-style random names to avoid mirroring
    the experiment name.

    Args:
        settings: A JobConfig instance.
        optuna_config: Optional Optuna configuration with study_name field.

    Returns:
        Study name for Optuna workflows.
    """
    run_name = settings.experiment.run_name if settings.experiment else None
    if isinstance(run_name, str) and run_name.strip():
        return run_name.strip()

    if settings.search is not None:
        study_name = settings.search.study_name
    else:
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
