"""Runtime-owned configuration validation entrypoint."""

from __future__ import annotations

import importlib.util
from typing import cast

from dlkit.common.errors import WorkflowError
from dlkit.engine.workflows.factories.build_factory import BuildFactory
from dlkit.infrastructure.config.protocols import BaseSettingsProtocol
from dlkit.infrastructure.config.workflow_configs import (
    InferenceWorkflowConfig,
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)


def validate_config(settings: BaseSettingsProtocol, dry_build: bool = False) -> bool:
    """Validate configuration structure and optional runtime readiness."""

    def structurally_valid(config: BaseSettingsProtocol) -> tuple[bool, str | None]:
        if not config.MODEL:
            return False, "[MODEL] section is required"
        if not config.DATASET:
            return False, "[DATASET] section is required"
        if not config.DATAMODULE:
            return False, "[DATAMODULE] section is required"

        # Guard clause: inference-only validation
        is_inference = isinstance(config, InferenceWorkflowConfig)
        if is_inference:
            if not (config.MODEL and config.MODEL.checkpoint):
                return False, "[MODEL.checkpoint] is required for inference mode"
            return True, None

        # Guard clause: training/optimization validation
        if not getattr(config, "TRAINING", None):
            return False, "[TRAINING] section is required for training"
        return True, None

    try:
        valid, message = structurally_valid(settings)
        if valid and getattr(settings, "MLFLOW", None) is not None:
            if importlib.util.find_spec("mlflow") is None:
                valid, message = False, "MLflow is not installed"

        # Check Optuna availability for optimization workflows
        if valid and isinstance(settings, OptimizationWorkflowConfig):
            if importlib.util.find_spec("optuna") is None:
                valid, message = False, "Optuna is not installed"

        if valid and dry_build:
            # Only build components for training/optimization workflows, not inference
            if not isinstance(settings, InferenceWorkflowConfig):
                try:
                    BuildFactory().build_components(
                        cast(TrainingWorkflowConfig | OptimizationWorkflowConfig, settings)
                    )
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
