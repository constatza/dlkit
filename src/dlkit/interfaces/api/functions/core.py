"""Core API functions for training, optimization, and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.interfaces.api.commands import (
    TrainCommandInput,
    OptimizationCommandInput,
    get_dispatcher,
)
from dlkit.interfaces.api.domain import (
    TrainingResult,
    InferenceResult,
    OptimizationResult,
)
from dlkit.tools.config.protocols import BaseSettingsProtocol
from dlkit.tools.config.workflow_settings import TrainingWorkflowSettings

# BREAKING CHANGE: Import new inference API
from dlkit.interfaces.inference.api import (
    infer as inference_api_infer,
    predict,
    InferenceInput,
)

# Get the global command dispatcher
_dispatcher = get_dispatcher()


def train(
    settings: BaseSettingsProtocol,
    mlflow: bool = False,
    checkpoint_path: Path | str | None = None,
    # Root override
    root_dir: Path | str | None = None,
    # Basic overrides
    output_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
    # Training overrides
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    # MLflow overrides
    mlflow_host: str | None = None,
    mlflow_port: int | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    **additional_overrides: Any,
) -> TrainingResult:
    """Run training with optional overrides.

    Note: For optimization (Optuna), use `optimize()` instead.
    This function specifically performs training workflows only.

    Args:
        settings: Parsed and validated configuration.
        mlflow: Enable MLflow tracking (overrides config settings).
        checkpoint_path: Checkpoint to resume from (overrides `[MODEL].checkpoint`).
        root_dir: Override the root directory for path resolution.
        output_dir: Override the output base directory.
        data_dir: Override the input dataflow directory.
        epochs: Override `[TRAINING].epochs` and `[TRAINING.trainer].max_epochs`.
        batch_size: Override `[DATAMODULE].batch_size` and `[DATAMODULE.dataloader].batch_size`.
        learning_rate: Override `[TRAINING.optimizer].lr`.
        mlflow_host: Override `[MLFLOW.server].host`.
        mlflow_port: Override `[MLFLOW.server].port`.
        experiment_name: Override `[MLFLOW.client].experiment_name`.
        run_name: Override `[MLFLOW.client].run_name`.
        additional_overrides: Extra overrides passed through to the manager.

    Returns:
        TrainingResult: Training execution result.

    Raises:
        WorkflowError: On execution failure.

    Example:
        >>> from dlkit.interfaces.api import train, validate_config
        >>> from dlkit.tools.config import load_training_settings
        >>> settings = load_training_settings("config.toml")  # Optimized for training workflows
        >>> validate_config(settings, dry_build=True)
        True
        >>> result = train(settings, mlflow=True, epochs=20, batch_size=64)
        >>> print(result.metrics)
    """
    input_data = TrainCommandInput(
        mlflow=mlflow,
        checkpoint_path=checkpoint_path,
        root_dir=root_dir,
        output_dir=output_dir,
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mlflow_host=mlflow_host,
        mlflow_port=mlflow_port,
        experiment_name=experiment_name,
        run_name=run_name,
        additional_overrides=additional_overrides,
    )

    return _dispatcher.execute("train", input_data, settings)


# BREAKING CHANGE: Replace old infer() function with new API

def infer(
    checkpoint_path: Path | str,
    inputs: Any,
    device: str = "auto",
    batch_size: int = 32,
    apply_transforms: bool = True
) -> InferenceResult:
    """Execute inference from checkpoint only.

    BREAKING CHANGE: This function now provides standalone inference
    that requires only a model checkpoint and input data. No training
    configuration files or datasets are needed.

    Args:
        checkpoint_path: Path to trained model checkpoint
        inputs: Input data (tensors, dict, arrays, or file path)
        device: Device specification ("auto", "cpu", "cuda", "mps")
        batch_size: Batch size for processing
        apply_transforms: Whether to apply fitted transforms

    Returns:
        InferenceResult: Inference execution result

    Raises:
        WorkflowError: On inference execution failure

    Example:
        >>> # New inference API
        >>> result = infer("model.ckpt", {"x": torch.randn(32, 10)})
        >>> predictions = result.predictions
        >>>
        >>> # For prediction mode with training config, use predict()
        >>> from dlkit.tools.config import load_training_settings
        >>> settings = load_training_settings("config.toml")
        >>> result = predict(settings, "model.ckpt")

    Migration:
        Old code:
            settings = load_inference_settings("config.toml")
            result = infer(settings, "model.ckpt")

        New code:
            result = infer("model.ckpt", your_input_data)
            # OR for prediction mode:
            settings = load_training_settings("config.toml")
            result = predict(settings, "model.ckpt")
    """
    return inference_api_infer(
        checkpoint_path=checkpoint_path,
        inputs=inputs,
        device=device,
        batch_size=batch_size,
        apply_transforms=apply_transforms
    )


# BREAKING CHANGE: Add new simple prediction function

def predict_with_config(
    training_settings: TrainingWorkflowSettings,
    checkpoint_path: Path | str,
    **overrides: Any
) -> InferenceResult:
    """Execute simple prediction using Lightning framework.

    This function provides the traditional Lightning-based inference approach
    for validation and testing scenarios where training configuration
    and datasets are available.

    Args:
        training_settings: Training workflow settings (BREAKING: InferenceWorkflowSettings no longer supported)
        checkpoint_path: Path to model checkpoint
        **overrides: Additional parameter overrides

    Returns:
        InferenceResult: Inference execution result

    Example:
        >>> from dlkit.tools.config import load_training_settings
        >>> settings = load_training_settings("config.toml")
        >>> result = predict_with_config(settings, "model.ckpt", batch_size=64)

    Note:
        This replaces the old infer() function when used with training configurations.
        For inference, use infer() directly.
    """
    return predict(
        training_settings=training_settings,
        checkpoint_path=checkpoint_path,
        **overrides
    )


def optimize(
    settings: BaseSettingsProtocol,
    trials: int = 100,
    mlflow: bool = False,
    checkpoint_path: Path | str | None = None,
    # Root override
    root_dir: Path | str | None = None,
    # Basic overrides
    output_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
    # Optuna overrides
    study_name: str | None = None,
    # MLflow overrides
    experiment_name: str | None = None,
    run_name: str | None = None,
    **additional_overrides: Any,
) -> OptimizationResult:
    """Run Optuna hyperparameter optimization with optional overrides.

    Args:
        settings: Parsed and validated configuration (with `[OPTUNA].enabled=true`).
        trials: Number of trials to run.
        mlflow: Enable MLflow tracking (overrides config settings).
        checkpoint_path: Optional warm-start checkpoint path.
        root_dir: Override the root directory for path resolution.
        output_dir: Override the output base directory.
        data_dir: Override the input dataflow directory.
        study_name: Override `[OPTUNA].study_name`.
        experiment_name: Override `[MLFLOW.client].experiment_name`.
        run_name: Override `[MLFLOW.client].run_name`.
        additional_overrides: Extra overrides passed through to the manager.

    Returns:
        OptimizationResult: Optimization summary, best trial, and final training result.

    Raises:
        WorkflowError: On execution failure.

    Example:
        >>> from dlkit.interfaces.api import optimize
        >>> from dlkit.tools.config import load_training_settings
        >>> settings = load_training_settings("optuna_config.toml")  # Loads OPTUNA section
        >>> result = optimize(settings, trials=50, mlflow=True, study_name="study")
        >>> result.best_trial
    """
    input_data = OptimizationCommandInput(
        trials=trials,
        mlflow=mlflow,
        checkpoint_path=checkpoint_path,
        root_dir=root_dir,
        output_dir=output_dir,
        data_dir=data_dir,
        study_name=study_name,
        experiment_name=experiment_name,
        run_name=run_name,
        additional_overrides=additional_overrides,
    )

    return _dispatcher.execute("optimize", input_data, settings)
