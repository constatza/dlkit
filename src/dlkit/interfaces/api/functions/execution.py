"""Unified execution function with intelligent workflow routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.interfaces.api.domain import (
    TrainingResult,
    InferenceResult,
    OptimizationResult,
)
from dlkit.interfaces.api.services.execution_service import ExecutionService
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.protocols import BaseSettingsProtocol
from dlkit.tools.config.workflow_configs import (
    TrainingWorkflowConfig,
    InferenceWorkflowConfig,
    OptimizationWorkflowConfig,
)


def execute(
    settings: (
        TrainingWorkflowConfig
        | InferenceWorkflowConfig
        | OptimizationWorkflowConfig
        | GeneralSettings
        | BaseSettingsProtocol
    ),
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
    """Execute DLKit workflow with intelligent routing based on settings.

    This unified function automatically determines the correct workflow based on configuration:

    **Priority-based workflow detection:**
    1. **Inference** (highest priority): when `settings.SESSION.inference=True`
    2. **Optimization**: when `settings.OPTUNA.enabled=True`
    3. **Training** (default): all other cases (includes MLflow if `settings.MLFLOW.enabled=True`)

    The system intelligently routes to the appropriate underlying service without requiring
    manual strategy specification, following SOLID principles for clean separation of concerns.

    Args:
        settings: Parsed and validated configuration settings (supports new workflow configs or legacy GeneralSettings)
        mlflow: Enable MLflow tracking (overrides config settings)
        checkpoint_path: Optional checkpoint path (required for inference, optional for others)
        root_dir: Override the root directory for path resolution
        output_dir: Override the output base directory
        data_dir: Override the input dataflow directory
        epochs: Override training epochs (applied to training and optimization workflows)
        batch_size: Override batch size (applied to all workflows)
        learning_rate: Override learning rate (applied to training and optimization workflows)
        trials: Override number of optimization trials (applied to optimization workflows)
        study_name: Override Optuna study name (applied to optimization workflows)
        experiment_name: Override MLflow experiment name (applied when MLflow enabled)
        run_name: Override MLflow run name (applied when MLflow enabled)
        **additional_overrides: Extra overrides passed to underlying services

    Returns:
        Appropriate result type based on detected workflow:
        - InferenceResult: for inference workflows
        - OptimizationResult: for optimization workflows (includes best trial info)
        - TrainingResult: for training workflows

    Raises:
        WorkflowError: On execution failure or invalid configuration

    Examples:
        >>> from dlkit.interfaces.api import execute
        >>> from dlkit.tools.config import GeneralSettings
        >>>
        >>> # Training with MLflow (auto-detected from settings)
        >>> from dlkit.tools.config import load_training_settings
        >>> settings = load_training_settings("training_config.toml")  # Faster loading
        >>> result = execute(settings, mlflow=True, epochs=50, batch_size=32)
        >>> print(f"Training loss: {result.metrics['train_loss']}")
        >>>
        >>> # Optimization with Optuna (auto-detected from settings)
        >>> optuna_settings = load_training_settings("optuna_config.toml")  # Includes OPTUNA
        >>> result = execute(optuna_settings, mlflow=True, trials=100, study_name="my_study")
        >>> print(f"Best trial: {result.best_trial}")
        >>>
        >>> # Inference (auto-detected from settings.SESSION.inference=True)
        >>> from dlkit.tools.config import load_inference_settings
        >>> inference_settings = load_inference_settings("inference_config.toml")  # Much faster
        >>> result = execute(inference_settings, checkpoint_path="best_model.ckpt")
        >>> predictions = result.predictions
    """
    execution_service = ExecutionService()
    return execution_service.execute(
        settings=settings,
        mlflow=mlflow,
        checkpoint_path=checkpoint_path,
        root_dir=root_dir,
        output_dir=output_dir,
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        trials=trials,
        study_name=study_name,
        experiment_name=experiment_name,
        run_name=run_name,
        **additional_overrides,
    )
