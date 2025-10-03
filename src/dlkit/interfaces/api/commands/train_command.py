"""Train command implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from dlkit.interfaces.api.domain.errors import WorkflowError
from dlkit.interfaces.api.domain.models import TrainingResult
from dlkit.interfaces.api.services import TrainingService
from dlkit.interfaces.api.overrides import basic_override_manager
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.protocols import BaseSettingsProtocol
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

    def validate_input(self, input_data: TrainCommandInput, settings: BaseSettingsProtocol) -> None:
        """Validate train command input.

        Args:
            input_data: Training parameters and overrides
            settings: DLKit configuration

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
            logger.error(error_msg, command="train", error_type=type(e).__name__, exc_info=True)
            raise WorkflowError(error_msg, {"command": "train", "error": str(e)}) from e

    def execute(
        self, input_data: TrainCommandInput, settings: BaseSettingsProtocol, **kwargs: Any
    ) -> TrainingResult:
        """Execute training command.

        Args:
            input_data: Training parameters and overrides
            settings: DLKit configuration
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
                f"Training execution failed: {e.message}",
                command="train",
                context=e.context,
            )
            raise
        except Exception as e:
            error_msg = f"Training execution failed: {str(e)}"
            logger.error(
                error_msg,
                command="train",
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise WorkflowError(
                error_msg, {"command": "train", "error_type": type(e).__name__}
            ) from e

    def _build_overrides_dict(self, input_data: TrainCommandInput) -> dict[str, Any]:
        """Build overrides dictionary from input

        Args:
            input_data: Training command input

        Returns:
            Dictionary of non-None overrides
        """
        # Build base overrides (None values filtered out)
        overrides = {
            k: v
            for k, v in {
                "checkpoint_path": (
                    Path(input_data.checkpoint_path)
                    if isinstance(input_data.checkpoint_path, str)
                    else input_data.checkpoint_path
                ),
                "root_dir": (
                    Path(input_data.root_dir)
                    if isinstance(input_data.root_dir, str)
                    else input_data.root_dir
                ),
                "output_dir": (
                    Path(input_data.output_dir)
                    if isinstance(input_data.output_dir, str)
                    else input_data.output_dir
                ),
                "data_dir": (
                    Path(input_data.data_dir)
                    if isinstance(input_data.data_dir, str)
                    else input_data.data_dir
                ),
                "epochs": input_data.epochs,
                "batch_size": input_data.batch_size,
                "learning_rate": input_data.learning_rate,
                "mlflow_host": input_data.mlflow_host,
                "mlflow_port": input_data.mlflow_port,
                "experiment_name": input_data.experiment_name,
                "run_name": input_data.run_name,
                **input_data.additional_overrides,
            }.items()
            if v is not None
        }

        # Always include mlflow boolean since it's always meaningful
        overrides["mlflow"] = input_data.mlflow

        return overrides
