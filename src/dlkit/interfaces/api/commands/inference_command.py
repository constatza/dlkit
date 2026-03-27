"""Inference command implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dlkit.interfaces.api.commands.normalizer import OverrideNormalizer
from dlkit.interfaces.api.domain.errors import WorkflowError
from dlkit.interfaces.api.domain.models import InferenceResult
from dlkit.interfaces.api.services import InferenceService, basic_override_manager
from dlkit.tools.config.workflow_configs import InferenceWorkflowConfig

from .base import BaseCommand


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceCommandInput:
    """Input dataflow for inference command."""

    checkpoint_path: Path | str  # Required
    root_dir: Path | str | None = None
    # Basic overrides
    output_dir: Path | str | None = None
    data_dir: Path | str | None = None
    batch_size: int | None = None
    # Additional overrides
    additional_overrides: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Initialize additional_overrides if None."""
        if self.additional_overrides is None:
            object.__setattr__(self, "additional_overrides", {})


class InferenceCommand(
    BaseCommand[InferenceCommandInput, InferenceResult, InferenceWorkflowConfig]
):
    """Command for executing inference workflows.

    Handles inference workflow execution with checkpoint loading,
    parameter overrides, and proper error handling.
    """

    def __init__(self, command_name: str = "infer") -> None:
        """Initialize inference command."""
        super().__init__(command_name)
        self.override_manager = basic_override_manager
        self.inference_service = InferenceService()

    def validate_input(
        self,
        input_data: InferenceCommandInput,
        settings: InferenceWorkflowConfig,
    ) -> None:
        """Validate inference command input.

        Args:
            input_data: Inference parameters and overrides
            settings: DLKit configuration (supports new InferenceWorkflowConfig or legacy GeneralSettings)

        Raises:
            WorkflowError: On validation failure
        """
        try:
            # Validate checkpoint path is provided
            if not input_data.checkpoint_path:
                raise WorkflowError(
                    "Checkpoint path is required for inference",
                    {"command": "infer", "error": "missing_checkpoint_path"},
                )

            # Build overrides dictionary
            overrides = self._build_overrides_dict(input_data)

            # Validate overrides
            if overrides:
                validation_errors = self.override_manager.validate_overrides(settings, **overrides)
                if validation_errors:
                    raise WorkflowError(
                        f"Override validation failed: {'; '.join(validation_errors)}",
                        {"command": "infer", "validation_errors": "; ".join(validation_errors)},
                    )

        except WorkflowError:
            raise
        except Exception as e:
            raise WorkflowError(
                f"Input validation failed: {e!s}", {"command": "infer", "error": str(e)}
            ) from e

    def execute(
        self,
        input_data: InferenceCommandInput,
        settings: InferenceWorkflowConfig,
        **kwargs: Any,
    ) -> InferenceResult:
        """Execute inference command.

        Args:
            input_data: Inference parameters and overrides
            settings: DLKit configuration (supports new InferenceWorkflowConfig or legacy GeneralSettings)
            **kwargs: Additional parameters

        Returns:
            InferenceResult on successful execution

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

            # Execute inference
            checkpoint_path = overrides.get("checkpoint_path", Path(input_data.checkpoint_path))
            return self.inference_service.infer(settings, checkpoint_path)

        except WorkflowError:
            raise
        except Exception as e:
            raise WorkflowError(
                f"Inference execution failed: {e!s}",
                {"command": "infer", "error_type": type(e).__name__},
            ) from e

    def _build_overrides_dict(self, input_data: InferenceCommandInput) -> dict[str, Any]:
        """Build overrides dictionary from input using OverrideNormalizer.

        Delegates to OverrideNormalizer.build_overrides_dict() for automatic
        path normalization and None-filtering. This eliminates duplication
        across command classes.

        Args:
            input_data: Inference command input

        Returns:
            Dictionary of non-None overrides with paths normalized to Path objects
        """
        return OverrideNormalizer.build_overrides_dict(
            checkpoint_path=input_data.checkpoint_path,
            root_dir=input_data.root_dir,
            output_dir=input_data.output_dir,
            data_dir=input_data.data_dir,
            batch_size=input_data.batch_size,
            additional_overrides=input_data.additional_overrides,
        )
