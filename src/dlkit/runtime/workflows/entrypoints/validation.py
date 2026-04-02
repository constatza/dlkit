"""Runtime-owned configuration validation entrypoint."""

from __future__ import annotations

from typing import cast

from dlkit.runtime.workflows.factories.build_factory import BuildFactory
from dlkit.shared.errors import WorkflowError
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.protocols import BaseSettingsProtocol
from dlkit.tools.config.workflow_configs import OptimizationWorkflowConfig, TrainingWorkflowConfig


def validate_config(settings: BaseSettingsProtocol, dry_build: bool = False) -> bool:
    """Validate configuration structure and optional runtime readiness."""

    def structurally_valid(config: BaseSettingsProtocol) -> tuple[bool, str | None]:
        if not config.MODEL:
            return False, "[MODEL] section is required"
        if not config.DATASET:
            return False, "[DATASET] section is required"
        if not config.DATAMODULE:
            return False, "[DATAMODULE] section is required"
        if not (config.SESSION and getattr(config.SESSION, "inference", False)):
            if not getattr(config, "TRAINING", None):
                return False, "[TRAINING] section is required for training"
        if config.SESSION and getattr(config.SESSION, "inference", False):
            if not (config.MODEL and config.MODEL.checkpoint):
                return False, "[MODEL.checkpoint] is required for inference mode"
        return True, None

    try:
        valid, message = structurally_valid(settings)
        if valid and getattr(settings, "MLFLOW", None) is not None:
            try:
                import mlflow  # noqa: F401
            except Exception as exc:
                valid, message = False, f"MLflow not available: {exc}"
        if valid and getattr(getattr(settings, "OPTUNA", None), "enabled", False):
            try:
                import optuna  # noqa: F401
            except Exception as exc:
                valid, message = False, f"Optuna not available: {exc}"
        if valid and dry_build:
            try:
                BuildFactory().build_components(
                    cast(
                        GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
                        settings,
                    )
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
