"""Runtime-owned configuration validation entrypoint."""

from __future__ import annotations

import importlib.util
from typing import cast

from dlkit.common.errors import WorkflowError
from dlkit.engine.workflows.factories.build_factory import BuildFactory
from dlkit.infrastructure.config.job_config import (
    InferenceJobConfig,
    JobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)


def validate_config(
    settings: TrainingJobConfig | SearchJobConfig | InferenceJobConfig | JobConfig,
    dry_build: bool = False,
) -> bool:
    """Validate configuration structure and optional runtime readiness."""

    def structurally_valid(
        config: TrainingJobConfig | SearchJobConfig | InferenceJobConfig | JobConfig,
    ) -> tuple[bool, str | None]:
        if config.model is None:
            return False, "[model] section is required"
        if config.data is None and not isinstance(config, InferenceJobConfig):
            return False, "[data] section is required"

        if isinstance(config, InferenceJobConfig):
            if config.model.checkpoint is None:
                return False, "[model.checkpoint] is required for inference mode"
            return True, None

        if isinstance(config, (TrainingJobConfig, SearchJobConfig)):
            if config.training is None:
                return False, "[training] section is required for training"
        return True, None

    try:
        valid, message = structurally_valid(settings)
        if valid and settings.tracking.backend == "mlflow":
            if importlib.util.find_spec("mlflow") is None:
                valid, message = False, "MLflow is not installed"

        if valid and isinstance(settings, SearchJobConfig):
            if importlib.util.find_spec("optuna") is None:
                valid, message = False, "Optuna is not installed"

        if valid and dry_build and not isinstance(settings, InferenceJobConfig):
            try:
                BuildFactory().build_components(cast(TrainingJobConfig | SearchJobConfig, settings))
            except WorkflowError:
                raise
            except Exception as exc:
                valid, message = False, f"Dry build failed: {exc}"

        if not valid:
            raise WorkflowError(
                f"Configuration validation failed: {message}",
                {"workflow": "validation"},
            )
        return True
    except WorkflowError:
        raise
    except Exception as exc:
        raise WorkflowError(
            f"Validation execution failed: {exc!s}",
            {"workflow": "validation", "error_type": type(exc).__name__},
        ) from exc
