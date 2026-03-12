"""Execution service for training-family workflows."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from dataclasses import dataclass

from dlkit.interfaces.api.domain import (
    TrainingResult,
    OptimizationResult,
    WorkflowError,
)
from dlkit.interfaces.api.tracking_hooks import TrackingHooks
from dlkit.tools.config import GeneralSettings
from dlkit.tools.utils.logging_config import get_logger

from .training_service import TrainingService
from .optimization_service import OptimizationService

logger = get_logger(__name__, "execution_service")


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowDetectionResult:
    """Result of training-family workflow detection logic."""

    workflow_type: str  # "optimization" | "training"
    service_class: type
    reasoning: str
    optuna_enabled: bool


class ExecutionService:
    """Execution service for training-family workflows.

    This service implements workflow detection following these priorities:
    1. Optimization mode - when settings.OPTUNA.enabled=True
    2. Training mode (default) - all other cases

    Follows SOLID principles:
    - SRP: Single responsibility is training-family workflow routing
    - OCP: New workflows can be added by extending detection logic
    - LSP: All underlying services maintain their contracts
    - ISP: Clean interface without workflow-specific bloat
    - DIP: Depends on service abstractions, not concrete implementations
    """

    def __init__(self) -> None:
        """Initialize execution service with all underlying services."""
        self.training_service = TrainingService()
        self.optimization_service = OptimizationService()
        self.service_name = "execution_service"

    def execute(
        self,
        settings: GeneralSettings,
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
        tags: dict[str, str] | None = None,
        hooks: TrackingHooks | None = None,
        **additional_overrides: Any,
    ) -> TrainingResult | OptimizationResult:
        """Execute workflow with intelligent routing based on settings.

        Automatically determines the correct workflow based on configuration:
        - Optimization: when settings.OPTUNA.enabled=True
        - Training: default case

        Args:
            settings: DLKit configuration settings
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
            tags: Key-value tags attached to every MLflow run
            hooks: Functional extension points for tracking lifecycle events
            **additional_overrides: Extra overrides passed to underlying services

        Returns:
            Appropriate result type based on detected workflow

        Raises:
            WorkflowError: On execution failure, invalid configuration,
                or when inference settings are passed to the training-family API
        """
        start_time = time.time()

        try:
            self._ensure_not_inference(settings)
            detection = self._detect_workflow(settings)

            logger.info(
                f"Detected workflow: {detection.workflow_type}",
                reasoning=detection.reasoning,
                optuna_enabled=detection.optuna_enabled,
            )

            if detection.workflow_type == "optimization":
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
                    tags=tags,
                    hooks=hooks,
                    **additional_overrides,
                )

        except Exception as e:
            duration = time.time() - start_time
            if isinstance(e, WorkflowError):
                raise
            raise WorkflowError(
                f"Execution service failed: {str(e)}",
                {
                    "service": self.service_name,
                    "duration_seconds": str(duration),
                    "error": str(e),
                },
            ) from e

    def _detect_workflow(
        self,
        settings: GeneralSettings,
    ) -> WorkflowDetectionResult:
        """Detect training-family workflow type based on settings.

        Priority order:
        1. Optimization mode
        2. Training mode (default)
        """
        if self._is_optuna_enabled(settings):
            return WorkflowDetectionResult(
                workflow_type="optimization",
                service_class=OptimizationService,
                reasoning="settings.OPTUNA.enabled=True",
                optuna_enabled=True,
            )

        return WorkflowDetectionResult(
            workflow_type="training",
            service_class=TrainingService,
            reasoning="default workflow (no optimization or inference flags)",
            optuna_enabled=False,
        )

    def _ensure_not_inference(self, settings: GeneralSettings) -> None:
        """Reject inference settings for the unified training-family execution API."""
        session = getattr(settings, "SESSION", None)
        if session and getattr(session, "inference", False):
            raise WorkflowError(
                "execute() no longer supports inference workflows. "
                "Use dlkit.load_model() or the dedicated inference API instead.",
                {"service": self.service_name, "workflow": "inference"},
            )

    def _is_optuna_enabled(self, settings: GeneralSettings) -> bool:
        """Check if Optuna optimization is enabled in settings."""
        optuna_config = getattr(settings, "OPTUNA", None)
        return bool(optuna_config and getattr(optuna_config, "enabled", False))

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

        final_overrides = {**opt_overrides, **additional_overrides}

        if final_overrides:
            settings = basic_override_manager.apply_overrides(settings, **final_overrides)

        final_checkpoint_path = final_overrides.get("checkpoint_path")

        return self.optimization_service.execute_optimization(
            settings=settings,
            trials=trials or 100,
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
        tags: dict[str, str] | None = None,
        hooks: TrackingHooks | None = None,
        **additional_overrides: Any,
    ) -> TrainingResult:
        """Execute training workflow with intelligent parameter handling."""
        training_overrides: dict[str, Any] = {}
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
        if tags is not None:
            training_overrides["tags"] = tags

        all_overrides = {**training_overrides, **additional_overrides}

        from dlkit.interfaces.api.overrides import basic_override_manager

        path_overrides: dict[str, Any] = {}
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

        if final_overrides:
            settings = basic_override_manager.apply_overrides(settings, **final_overrides)

        final_checkpoint_path = final_overrides.get("checkpoint_path")

        return self.training_service.execute_training(
            settings=settings,
            checkpoint_path=final_checkpoint_path,
            hooks=hooks,
        )
