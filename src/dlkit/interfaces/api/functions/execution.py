"""Unified execution function with intelligent workflow routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from dlkit.interfaces.api.domain import (
    OptimizationResult,
    TrainingResult,
)
from dlkit.interfaces.api.services.execution_service import ExecutionService
from dlkit.interfaces.api.tracking_hooks import TrackingHooks
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.protocols import BaseSettingsProtocol
from dlkit.tools.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)


def execute(
    settings: (
        TrainingWorkflowConfig | OptimizationWorkflowConfig | GeneralSettings | BaseSettingsProtocol
    ),
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
    """Execute DLKit workflow with intelligent routing based on settings.

    This unified function automatically determines the correct training-family
    workflow based on configuration:

    **Workflow detection:**
    1. **Optimization**: when `settings.OPTUNA.enabled=True`
    2. **Training** (default): all other cases

    Args:
        settings: Parsed and validated configuration settings
        checkpoint_path: Optional training resume or optimization warm-start checkpoint
        root_dir: Override the root directory for path resolution
        output_dir: Override the output base directory
        data_dir: Override the input dataflow directory
        epochs: Override training epochs (applied to training and optimization workflows)
        batch_size: Override batch size (applied to all workflows)
        learning_rate: Override learning rate (applied to training and optimization workflows)
        trials: Override number of optimization trials (applied to optimization workflows)
        study_name: Override Optuna study name (applied to optimization workflows)
        experiment_name: Override MLflow experiment name
        run_name: Override MLflow run name
        tags: Key-value tags attached to every MLflow run (merged with settings tags)
        hooks: Functional extension points for tracking lifecycle events
        **additional_overrides: Extra overrides passed to underlying services

    Returns:
        Appropriate result type based on detected workflow:
        - OptimizationResult: for optimization workflows (includes best trial info)
        - TrainingResult: for training workflows

    Raises:
        WorkflowError: On execution failure, invalid configuration, or when
            inference settings are passed to this training-family API

    Examples:
        >>> from dlkit.interfaces.api import execute, TrackingHooks
        >>> from dlkit.tools.io import load_settings
        >>>
        >>> # Training with MLflow (auto-detected from settings)
        >>> settings = load_settings("training_config.toml")
        >>> result = execute(settings, epochs=50, batch_size=32)
        >>> result.mlflow_run_id is not None
        True
        >>>
        >>> # With hooks
        >>> result = execute(
        ...     settings,
        ...     checkpoint_path="resume.ckpt",
        ...     tags={"team": "ml", "release": "spring"},
        ...     hooks=TrackingHooks(
        ...         on_run_created=lambda run_id, uri: None,
        ...     ),
        ... )
        >>>
        >>> # Inference is separate - use load_model()
        >>> from dlkit.interfaces.inference import load_model
        >>> predictor = load_model("best_model.ckpt")
        >>> predictions = predictor.predict(inputs)
    """
    execution_service = ExecutionService()
    return execution_service.execute(
        settings=cast(GeneralSettings, settings),
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
        tags=tags,
        hooks=hooks,
        **additional_overrides,
    )
