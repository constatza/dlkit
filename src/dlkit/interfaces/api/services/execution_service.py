"""Intelligent execution service that automatically determines workflow based on settings."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from dataclasses import dataclass

from dlkit.interfaces.api.domain import (
    TrainingResult,
    InferenceResult,
    OptimizationResult,
    WorkflowError,
)
from dlkit.tools.config import GeneralSettings
from dlkit.tools.utils.logging_config import get_logger

from .training_service import TrainingService
from .inference_service import InferenceService
from .optimization_service import OptimizationService

logger = get_logger(__name__, "execution_service")


@dataclass(frozen=True)
class WorkflowDetectionResult:
    """Result of workflow detection logic."""

    workflow_type: str  # "inference", "optimization", "training"
    service_class: type
    reasoning: str
    mlflow_enabled: bool
    optuna_enabled: bool


class ExecutionService:
    """Intelligent execution service that automatically determines workflow based on settings.

    This service implements intelligent workflow detection following these priorities:
    1. Inference mode (highest priority) - when settings.SESSION.inference=True
    2. Optimization mode - when settings.OPTUNA.enabled=True
    3. Training mode (default) - all other cases

    Follows SOLID principles:
    - SRP: Single responsibility is intelligent workflow routing
    - OCP: New workflows can be added by extending detection logic
    - LSP: All underlying services maintain their contracts
    - ISP: Clean interface without workflow-specific bloat
    - DIP: Depends on service abstractions, not concrete implementations
    """

    def __init__(self) -> None:
        """Initialize execution service with all underlying services."""
        self.training_service = TrainingService()
        self.inference_service = InferenceService()
        self.optimization_service = OptimizationService()
        self.service_name = "execution_service"

    def execute(
        self,
        settings: GeneralSettings,
        mlflow: bool = False,
        checkpoint_path: Path | str | None = None,
        root_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
        data_dir: Path | str | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        trials: int | None = None,
        study_name: str | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
        **additional_overrides: Any,
    ) -> TrainingResult | InferenceResult | OptimizationResult:
        """Execute workflow with intelligent routing based on settings.

        Automatically determines the correct workflow based on configuration:
        - Inference: when settings.SESSION.inference=True
        - Optimization: when settings.OPTUNA.enabled=True
        - Training: default case (includes MLflow if settings.MLFLOW.enabled=True)

        Args:
            settings: DLKit configuration settings
            mlflow: Enable MLflow tracking (overrides config settings)
            checkpoint_path: Optional checkpoint path (may indicate inference intent)
            root_dir: Override the root directory for path resolution
            output_dir: Override the output base directory
            data_dir: Override the input dataflow directory
            epochs: Override training epochs
            batch_size: Override batch size
            learning_rate: Override learning rate
            trials: Override number of optimization trials
            study_name: Override Optuna study name
            experiment_name: Override MLflow experiment name
            run_name: Override MLflow run name
            **additional_overrides: Extra overrides passed to underlying services

        Returns:
            Appropriate result type based on detected workflow:
            - TrainingResult for training workflows
            - OptimizationResult for optimization workflows
            - InferenceResult for inference workflows

        Raises:
            WorkflowError: On execution failure or invalid configuration
        """
        start_time = time.time()

        try:
            # Detect workflow based on settings
            detection = self._detect_workflow(settings, checkpoint_path, mlflow)

            logger.info(
                f"Detected workflow: {detection.workflow_type}",
                reasoning=detection.reasoning,
                mlflow_enabled=detection.mlflow_enabled,
                optuna_enabled=detection.optuna_enabled,
            )

            # Route to appropriate service
            if detection.workflow_type == "inference":
                return self._execute_inference(
                    settings,
                    checkpoint_path,
                    root_dir,
                    output_dir,
                    data_dir,
                    batch_size,
                    **additional_overrides,
                )
            elif detection.workflow_type == "optimization":
                return self._execute_optimization(
                    settings,
                    trials,
                    checkpoint_path,
                    root_dir,
                    output_dir,
                    data_dir,
                    study_name,
                    experiment_name,
                    run_name,
                    **additional_overrides,
                )
            else:  # training
                return self._execute_training(
                    settings,
                    checkpoint_path,
                    root_dir,
                    output_dir,
                    data_dir,
                    epochs,
                    batch_size,
                    learning_rate,
                    experiment_name,
                    run_name,
                    **additional_overrides,
                )

        except Exception as e:
            duration = time.time() - start_time
            if isinstance(e, WorkflowError):
                raise
            raise WorkflowError(
                f"Execution service failed: {str(e)}",
                {"service": self.service_name, "duration_seconds": duration, "error": str(e)},
            ) from e

    def _detect_workflow(
        self,
        settings: GeneralSettings,
        checkpoint_path: Path | str | None,
        mlflow_override: bool = False,
    ) -> WorkflowDetectionResult:
        """Detect workflow type based on settings and parameters.

        Priority order:
        1. Inference mode (highest priority)
        2. Optimization mode
        3. Training mode (default)
        """
        # Check for explicit inference mode
        if settings.SESSION and getattr(settings.SESSION, "inference", False):
            return WorkflowDetectionResult(
                workflow_type="inference",
                service_class=InferenceService,
                reasoning="settings.SESSION.inference=True",
                mlflow_enabled=mlflow_override or self._is_mlflow_enabled(settings),
                optuna_enabled=False,  # Optuna not used in inference
            )

        # Check for optimization mode
        if self._is_optuna_enabled(settings):
            return WorkflowDetectionResult(
                workflow_type="optimization",
                service_class=OptimizationService,
                reasoning="settings.OPTUNA.enabled=True",
                mlflow_enabled=mlflow_override or self._is_mlflow_enabled(settings),
                optuna_enabled=True,
            )

        # Default to training mode
        return WorkflowDetectionResult(
            workflow_type="training",
            service_class=TrainingService,
            reasoning="default workflow (no optimization or inference flags)",
            mlflow_enabled=self._is_mlflow_enabled(settings),
            optuna_enabled=False,
        )

    def _is_mlflow_enabled(self, settings: GeneralSettings) -> bool:
        """Check if MLflow tracking is enabled in settings."""
        mlflow_config = getattr(settings, "MLFLOW", None)
        return bool(mlflow_config and getattr(mlflow_config, "enabled", False))

    def _is_optuna_enabled(self, settings: GeneralSettings) -> bool:
        """Check if Optuna optimization is enabled in settings."""
        optuna_config = getattr(settings, "OPTUNA", None)
        return bool(optuna_config and getattr(optuna_config, "enabled", False))

    def _execute_inference(
        self,
        settings: GeneralSettings,
        checkpoint_path: Path | str | None,
        root_dir: Path | str | None,
        output_dir: Path | str | None,
        data_dir: Path | str | None,
        batch_size: int | None,
        **additional_overrides: Any,
    ) -> InferenceResult:
        """Execute inference workflow with parameter validation."""
        if not checkpoint_path:
            raise WorkflowError(
                "Inference workflow requires checkpoint_path parameter",
                {"service": self.service_name, "workflow": "inference"},
            )

        # Apply overrides like the InferenceCommand does
        from dlkit.interfaces.api.overrides import basic_override_manager

        inference_overrides = {}
        if checkpoint_path is not None:
            inference_overrides["checkpoint_path"] = (
                Path(checkpoint_path) if isinstance(checkpoint_path, str) else checkpoint_path
            )
        if root_dir is not None:
            inference_overrides["root_dir"] = (
                Path(root_dir) if isinstance(root_dir, str) else root_dir
            )
        if output_dir is not None:
            inference_overrides["output_dir"] = (
                Path(output_dir) if isinstance(output_dir, str) else output_dir
            )
        if data_dir is not None:
            inference_overrides["data_dir"] = (
                Path(data_dir) if isinstance(data_dir, str) else data_dir
            )
        if batch_size is not None:
            inference_overrides["batch_size"] = batch_size

        final_overrides = {**inference_overrides, **additional_overrides}

        # Apply overrides to settings
        if final_overrides:
            settings = basic_override_manager.apply_overrides(settings, **final_overrides)

        # Extract final checkpoint path for service call
        final_checkpoint_path = final_overrides.get("checkpoint_path", checkpoint_path)

        return self.inference_service.execute_inference(
            settings=settings, checkpoint_path=final_checkpoint_path
        )

    def _execute_optimization(
        self,
        settings: GeneralSettings,
        trials: int | None,
        checkpoint_path: Path | str | None,
        root_dir: Path | str | None,
        output_dir: Path | str | None,
        data_dir: Path | str | None,
        study_name: str | None,
        experiment_name: str | None,
        run_name: str | None,
        **additional_overrides: Any,
    ) -> OptimizationResult:
        """Execute optimization workflow with intelligent parameter handling."""
        # Apply overrides like the OptimizationCommand does
        from dlkit.interfaces.api.overrides import basic_override_manager

        opt_overrides = {}
        if checkpoint_path is not None:
            opt_overrides["checkpoint_path"] = (
                Path(checkpoint_path) if isinstance(checkpoint_path, str) else checkpoint_path
            )
        if root_dir is not None:
            opt_overrides["root_dir"] = Path(root_dir) if isinstance(root_dir, str) else root_dir
        if output_dir is not None:
            opt_overrides["output_dir"] = (
                Path(output_dir) if isinstance(output_dir, str) else output_dir
            )
        if data_dir is not None:
            opt_overrides["data_dir"] = Path(data_dir) if isinstance(data_dir, str) else data_dir
        if study_name is not None:
            opt_overrides["study_name"] = study_name
        if experiment_name is not None:
            opt_overrides["experiment_name"] = experiment_name
        if run_name is not None:
            opt_overrides["run_name"] = run_name

        # Combine with additional overrides
        final_overrides = {**opt_overrides, **additional_overrides}

        # Apply overrides to settings
        if final_overrides:
            settings = basic_override_manager.apply_overrides(settings, **final_overrides)

        # Extract checkpoint path for service call
        final_checkpoint_path = final_overrides.get("checkpoint_path")

        return self.optimization_service.execute_optimization(
            settings=settings,
            trials=trials or 100,  # Default to 100 trials if not specified
            checkpoint_path=final_checkpoint_path,
        )

    def _execute_training(
        self,
        settings: GeneralSettings,
        checkpoint_path: Path | str | None,
        root_dir: Path | str | None,
        output_dir: Path | str | None,
        data_dir: Path | str | None,
        epochs: int | None,
        batch_size: int | None,
        learning_rate: float | None,
        experiment_name: str | None,
        run_name: str | None,
        **additional_overrides: Any,
    ) -> TrainingResult:
        """Execute training workflow with intelligent parameter handling."""
        # Build training-specific overrides
        training_overrides = {}
        if epochs is not None:
            training_overrides["epochs"] = epochs
        if batch_size is not None:
            training_overrides["batch_size"] = batch_size
        if learning_rate is not None:
            training_overrides["learning_rate"] = learning_rate
        if experiment_name is not None:
            training_overrides["experiment_name"] = experiment_name
        if run_name is not None:
            training_overrides["run_name"] = run_name

        # Combine with additional overrides
        all_overrides = {**training_overrides, **additional_overrides}

        # Apply path overrides via override manager (like the original train function does)
        from dlkit.interfaces.api.overrides import basic_override_manager

        path_overrides = {}
        if checkpoint_path is not None:
            path_overrides["checkpoint_path"] = (
                Path(checkpoint_path) if isinstance(checkpoint_path, str) else checkpoint_path
            )
        if root_dir is not None:
            path_overrides["root_dir"] = Path(root_dir) if isinstance(root_dir, str) else root_dir
        if output_dir is not None:
            path_overrides["output_dir"] = (
                Path(output_dir) if isinstance(output_dir, str) else output_dir
            )
        if data_dir is not None:
            path_overrides["data_dir"] = Path(data_dir) if isinstance(data_dir, str) else data_dir

        final_overrides = {**path_overrides, **all_overrides}

        # Apply overrides to settings
        if final_overrides:
            settings = basic_override_manager.apply_overrides(settings, **final_overrides)

        # Extract checkpoint path for service call
        final_checkpoint_path = final_overrides.get("checkpoint_path")

        return self.training_service.execute_training(
            settings=settings,
            checkpoint_path=final_checkpoint_path,
        )
