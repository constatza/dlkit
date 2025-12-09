"""Train command implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from dlkit.interfaces.api.domain.errors import WorkflowError
from dlkit.interfaces.api.domain.models import TrainingResult
from dlkit.interfaces.api.services import TrainingService
from dlkit.interfaces.api.overrides import OverrideNormalizer, basic_override_manager
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.protocols import BaseSettingsProtocol
from dlkit.tools.config.workflow_configs import TrainingWorkflowConfig
from .base import BaseCommand


@dataclass(frozen=True)
class TrainCommandInput:
    """Input dataflow for train command."""

    mlflow: bool = False
    checkpoint_path: Path | str | None = None
    root_dir: Path | str | None = None
    # Basic overrides
    output_dir: Path | str | None = None
    data_dir: Path | str | None = None
    # Training overrides
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    # MLflow overrides
    mlflow_host: str | None = None
    mlflow_port: int | None = None
    experiment_name: str | None = None
    run_name: str | None = None
    # Additional overrides
    additional_overrides: dict[str, Any] = field(default_factory=dict)


class TrainCommand(BaseCommand[TrainCommandInput, TrainingResult]):
    """Command for executing training workflows.

    Handles training workflow execution with parameter overrides,
    validation, and proper error handling. Follows the Command pattern
    for single responsibility and dependency injection.
    """

    def __init__(self, command_name: str = "train") -> None:
        """Initialize train command."""
        super().__init__(command_name)
        self.override_manager = basic_override_manager
        self.training_service = TrainingService()

    def validate_input(
        self,
        input_data: TrainCommandInput,
        settings: TrainingWorkflowConfig | GeneralSettings | BaseSettingsProtocol,
    ) -> None:
        """Validate train command input.

        Args:
            input_data: Training parameters and overrides
            settings: DLKit configuration (supports new TrainingWorkflowConfig or legacy GeneralSettings)

        Raises:
            WorkflowError: On validation failure
        """
        try:
            overrides = self._build_overrides_dict(input_data)
            if overrides:
                validation_errors = self.override_manager.validate_overrides(settings, **overrides)
                if validation_errors:
                    error_msg = f"Override validation failed: {'; '.join(validation_errors)}"
                    logger.error(
                        "{}",
                        error_msg,
                        command="train",
                        validation_errors=validation_errors,
                        overrides=overrides,
                    )
                    raise WorkflowError(
                        error_msg,
                        {"command": "train", "validation_errors": "; ".join(validation_errors)},
                    )
        except WorkflowError:
            raise
        except Exception as e:
            error_msg = f"Input validation failed: {str(e)}"
            logger.error(
                "{}",
                error_msg,
                command="train",
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise WorkflowError(error_msg, {"command": "train", "error": str(e)}) from e

    def execute(
        self,
        input_data: TrainCommandInput,
        settings: TrainingWorkflowConfig | GeneralSettings | BaseSettingsProtocol,
        **kwargs: Any,
    ) -> TrainingResult:
        """Execute training command.

        Args:
            input_data: Training parameters and overrides
            settings: DLKit configuration (supports new TrainingWorkflowConfig or legacy GeneralSettings)
            **kwargs: Additional parameters

        Returns:
            TrainingResult on successful execution

        Raises:
            WorkflowError: On execution failure
        """
        try:
            logger.info(
                "Starting training command execution",
                has_checkpoint=input_data.checkpoint_path is not None,
            )

            # Validate input
            self.validate_input(input_data, settings)

            # Build and apply overrides
            overrides = self._build_overrides_dict(input_data)
            if overrides:
                logger.debug("Applying training overrides", overrides=overrides)
                settings = self.override_manager.apply_overrides(settings, **overrides)

            # Execute training
            checkpoint = overrides.get("checkpoint_path") if overrides else None
            result = self.training_service.execute_training(
                settings, checkpoint
            )

            logger.info(
                "Training command completed successfully",
                duration_seconds=getattr(result, "duration_seconds", None),
            )

            return result

        except WorkflowError as e:
            logger.error(
                "Training execution failed: {}",
                e.message,
                command="train",
                context=e.context,
            )
            raise
        except Exception as e:
            error_msg = f"Training execution failed: {str(e)}"
            logger.error(
                "{}",
                error_msg,
                command="train",
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise WorkflowError(
                error_msg, {"command": "train", "error_type": type(e).__name__}
            ) from e

    def _build_overrides_dict(self, input_data: TrainCommandInput) -> dict[str, Any]:
        """Build overrides dictionary from input using OverrideNormalizer.

        Delegates to OverrideNormalizer.build_overrides_dict() for automatic
        path normalization and None-filtering. This eliminates duplication
        across command classes.

        Args:
            input_data: Training command input

        Returns:
            Dictionary of non-None overrides with paths normalized to Path objects
        """
        # Use shared normalizer for path normalization and None-filtering
        overrides = OverrideNormalizer.build_overrides_dict(
            checkpoint_path=input_data.checkpoint_path,
            root_dir=input_data.root_dir,
            output_dir=input_data.output_dir,
            data_dir=input_data.data_dir,
            epochs=input_data.epochs,
            batch_size=input_data.batch_size,
            learning_rate=input_data.learning_rate,
            mlflow_host=input_data.mlflow_host,
            mlflow_port=input_data.mlflow_port,
            experiment_name=input_data.experiment_name,
            run_name=input_data.run_name,
            additional_overrides=input_data.additional_overrides,
        )

        # Always include mlflow boolean since it's always meaningful
        overrides["mlflow"] = input_data.mlflow

        return overrides
