"""Optimization command implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dlkit.interfaces.api.commands.normalizer import OverrideNormalizer
from dlkit.interfaces.api.domain.errors import WorkflowError
from dlkit.interfaces.api.domain.models import OptimizationResult
from dlkit.interfaces.api.services import OptimizationService, basic_override_manager
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.workflow_configs import OptimizationWorkflowConfig

from .base import BaseCommand


@dataclass(frozen=True, slots=True, kw_only=True)
class OptimizationCommandInput:
    """Input dataflow for optimization command."""

    trials: int | None = None  # Override default should be None
    checkpoint_path: Path | str | None = None
    root_dir: Path | str | None = None
    # Optuna overrides
    study_name: str | None = None
    # MLflow overrides
    experiment_name: str | None = None
    run_name: str | None = None
    # Additional overrides
    additional_overrides: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Initialize additional_overrides if None."""
        if self.additional_overrides is None:
            object.__setattr__(self, "additional_overrides", {})


class OptimizationCommand(
    BaseCommand[
        OptimizationCommandInput, OptimizationResult, OptimizationWorkflowConfig | GeneralSettings
    ]
):
    """Command for executing hyperparameter optimization workflows.

    Handles Optuna-based optimization with parameter overrides,
    validation, and proper error handling.
    """

    def __init__(self, command_name: str = "optimize") -> None:
        """Initialize optimization command."""
        super().__init__(command_name)
        self.override_manager = basic_override_manager
        self.optimization_service = OptimizationService()

    def validate_input(
        self,
        input_data: OptimizationCommandInput,
        settings: OptimizationWorkflowConfig | GeneralSettings,
    ) -> None:
        """Validate optimization command input.

        Args:
            input_data: Optimization parameters and overrides
            settings: DLKit configuration (supports OptimizationWorkflowConfig or legacy GeneralSettings)

        Raises:
            WorkflowError: On validation failure
        """
        try:
            # Build overrides dictionary
            overrides = self._build_overrides_dict(input_data)

            # Validate overrides
            if overrides:
                validation_errors = self.override_manager.validate_overrides(settings, **overrides)
                if validation_errors:
                    raise WorkflowError(
                        f"Override validation failed: {'; '.join(validation_errors)}",
                        {"command": "optimize", "validation_errors": "; ".join(validation_errors)},
                    )

        except WorkflowError:
            raise
        except Exception as e:
            raise WorkflowError(
                f"Input validation failed: {e!s}", {"command": "optimize", "error": str(e)}
            ) from e

    def execute(
        self,
        input_data: OptimizationCommandInput,
        settings: OptimizationWorkflowConfig | GeneralSettings,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Execute optimization command.

        Args:
            input_data: Optimization parameters and overrides
            settings: DLKit configuration (supports OptimizationWorkflowConfig or legacy GeneralSettings)
            **kwargs: Additional parameters

        Returns:
            OptimizationResult on successful execution

        Raises:
            WorkflowError: On execution failure
        """
        try:
            # Validate input
            self.validate_input(input_data, settings)

            # Build and apply overrides
            overrides = self._build_overrides_dict(input_data)
            if overrides:
                settings = self.override_manager.apply_overrides(settings, **overrides)

            # Execute optimization — GeneralSettings is the supertype accepted by the service
            checkpoint = overrides.get("checkpoint_path") if overrides else None
            effective: GeneralSettings = (
                settings
                if isinstance(settings, GeneralSettings)
                else GeneralSettings.model_validate(settings.model_dump())
            )
            return self.optimization_service.execute_optimization(
                effective, input_data.trials or 100, checkpoint
            )

        except WorkflowError:
            raise
        except Exception as e:
            raise WorkflowError(
                f"Optimization execution failed: {e!s}",
                {"command": "optimize", "error_type": type(e).__name__},
            ) from e

    def _build_overrides_dict(self, input_data: OptimizationCommandInput) -> dict[str, Any]:
        """Build overrides dictionary from input using OverrideNormalizer.

        Delegates to OverrideNormalizer.build_overrides_dict() for automatic
        path normalization and None-filtering. This eliminates duplication
        across command classes.

        Args:
            input_data: Optimization command input

        Returns:
            Dictionary of non-None overrides with paths normalized to Path objects
        """
        return OverrideNormalizer.build_overrides_dict(
            checkpoint_path=input_data.checkpoint_path,
            root_dir=input_data.root_dir,
            trials=input_data.trials if input_data.trials != 100 else None,
            study_name=input_data.study_name,
            experiment_name=input_data.experiment_name,
            run_name=input_data.run_name,
            additional_overrides=input_data.additional_overrides,
        )
