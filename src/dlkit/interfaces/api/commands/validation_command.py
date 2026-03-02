"""Validation command implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dlkit.interfaces.api.domain.errors import WorkflowError
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.protocols import BaseSettingsProtocol
from .base import BaseCommand


@dataclass(frozen=True)
class ValidationCommandInput:
    """Input dataflow for validation command."""

    # If True, perform a dry build of components to catch runtime mismatches early
    dry_build: bool = False


class ValidationCommand(BaseCommand[ValidationCommandInput, bool]):
    """Command for validating configuration against strategies.

    Handles configuration validation with automatic strategy detection
    and detailed error reporting.
    """

    def __init__(self, command_name: str = "validate_config") -> None:
        """Initialize validation command."""
        super().__init__(command_name)

    def validate_input(
        self, input_data: ValidationCommandInput, settings: BaseSettingsProtocol
    ) -> None:
        """Validate validation command input.

        Args:
            input_data: Validation parameters
            settings: DLKit configuration

        Raises:
            WorkflowError: On validation failure
        """
        try:
            if not isinstance(input_data.dry_build, bool):
                raise ValueError("dry_build must be a boolean flag")

        except Exception as e:
            raise WorkflowError(
                f"Input validation failed: {str(e)}",
                {"command": "validate_config", "error": str(e)},
            ) from e

    def execute(
        self, input_data: ValidationCommandInput, settings: BaseSettingsProtocol, **kwargs: Any
    ) -> bool:
        """Execute validation command.

        Args:
            input_data: Validation parameters
            settings: DLKit configuration
            **kwargs: Additional parameters

        Returns:
            True if validation succeeds

        Raises:
            WorkflowError: On validation failure
        """
        profile = "unknown"
        try:
            # Validate input
            self.validate_input(input_data, settings)
            profile = self._describe_profile(settings)

            # Generic structural validation (strategy-agnostic)
            def _structurally_valid(s: BaseSettingsProtocol) -> tuple[bool, str | None]:
                # Require MODEL, DATASET, DATAMODULE in all modes
                if not s.MODEL:
                    return False, "[MODEL] section is required"
                if not s.DATASET:
                    return False, "[DATASET] section is required"
                if not s.DATAMODULE:
                    return False, "[DATAMODULE] section is required"
                # In training mode require TRAINING
                if not (s.SESSION and getattr(s.SESSION, "inference", False)):
                    if not s.TRAINING:
                        return False, "[TRAINING] section is required for training"
                # In inference mode require MODEL.checkpoint
                if s.SESSION and getattr(s.SESSION, "inference", False):
                    if not (s.MODEL and s.MODEL.checkpoint):
                        return False, "[MODEL.checkpoint] is required for inference mode"
                return True, None

            ok, msg = _structurally_valid(settings)

            # Optional environment checks for selected/active strategies
            if ok:
                try:
                    if self._mlflow_enabled(settings):
                        try:
                            import mlflow  # noqa: F401
                        except Exception as e:
                            ok = False
                            msg = f"MLflow not available: {e}"
                    if ok and self._optuna_enabled(settings):
                        try:
                            import optuna  # noqa: F401
                        except Exception as e:
                            ok = False
                            msg = f"Optuna not available: {e}"
                except Exception:
                    pass

            # Optional dry build to catch model/datamodule/shape issues
            if ok and input_data.dry_build:
                try:
                    from dlkit.runtime.workflows.factories.build_factory import BuildFactory

                    _ = BuildFactory().build_components(settings)
                except Exception as e:
                    ok = False
                    msg = f"Dry build failed: {e}"

            # Raise exception if validation failed
            if not ok:
                raise WorkflowError(
                    f"Configuration validation failed ({profile}): {msg}",
                    {"command": "validate_config", "profile": profile},
                )

            return True

        except WorkflowError:
            raise
        except Exception as e:
            raise WorkflowError(
                f"Validation execution failed: {str(e)}",
                {
                    "command": "validate_config",
                    "profile": profile,
                    "error_type": type(e).__name__,
                },
            ) from e

    def _describe_profile(self, settings: BaseSettingsProtocol) -> str:
        optim = self._optuna_enabled(settings)
        tracking = self._mlflow_enabled(settings)
        mode = "inference" if self._is_inference(settings) else "training"

        features: list[str] = [mode]
        if optim:
            features.append("optuna")
        if tracking:
            features.append("mlflow")

        return "+".join(features)

    def _optuna_enabled(self, settings: BaseSettingsProtocol) -> bool:
        optuna_cfg = getattr(settings, "OPTUNA", None)
        return bool(optuna_cfg and getattr(optuna_cfg, "enabled", False))

    def _mlflow_enabled(self, settings: BaseSettingsProtocol) -> bool:
        mlflow_cfg = getattr(settings, "MLFLOW", None)
        return bool(mlflow_cfg and getattr(mlflow_cfg, "enabled", False))

    def _is_inference(self, settings: BaseSettingsProtocol) -> bool:
        session = getattr(settings, "SESSION", None)
        return bool(session and getattr(session, "inference", False))
