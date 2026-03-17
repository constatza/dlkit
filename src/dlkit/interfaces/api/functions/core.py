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

# Inference API removed - use load_model() instead
# from dlkit import load_model
# predictor = load_model("model.ckpt")
# result = predictor.predict(inputs)

# Get the global command dispatcher
_dispatcher = get_dispatcher()


def train(
    settings: BaseSettingsProtocol,
    checkpoint_path: Path | str | None = None,
    # Root override
    root_dir: Path | str | None = None,
    # Training overrides
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    # MLflow overrides — only meaningful when settings already has an [MLFLOW] section
    experiment_name: str | None = None,
    run_name: str | None = None,
    **additional_overrides: Any,
) -> TrainingResult:
    """Run training with optional overrides.

    Note: For optimization (Optuna), use `optimize()` instead.
    This function specifically performs training workflows only.

    To enable MLflow tracking, include an ``[MLFLOW]`` section in your configuration
    (or pass ``GeneralSettings(MLFLOW=MLflowSettings(...))``) — there is no
    boolean ``mlflow`` toggle here.

    Args:
        settings: Parsed and validated configuration.
        checkpoint_path: Checkpoint to resume from (overrides ``[MODEL].checkpoint``).
        root_dir: Override the root directory for all path resolution.
        epochs: Override ``[TRAINING].epochs`` and ``[TRAINING.trainer].max_epochs``.
        batch_size: Override ``[DATAMODULE.dataloader].batch_size``.
        learning_rate: Override ``[TRAINING.optimizer].lr``.
        experiment_name: Override ``[MLFLOW].experiment_name``.
        run_name: Override ``[MLFLOW].run_name``.
        additional_overrides: Extra overrides passed through to the manager.

    Returns:
        TrainingResult: Training execution result.

    Raises:
        WorkflowError: On execution failure.
    """
    input_data = TrainCommandInput(
        checkpoint_path=checkpoint_path,
        root_dir=root_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        experiment_name=experiment_name,
        run_name=run_name,
        additional_overrides=additional_overrides,
    )

    return _dispatcher.execute("train", input_data, settings)


# REMOVED: Old infer() and predict_with_config() functions
# Use the new load_model() API instead:
#
#   from dlkit import load_model
#   predictor = load_model("model.ckpt", device="cuda")
#   result = predictor.predict(inputs)
#
# See documentation for migration guide.


def optimize(
    settings: BaseSettingsProtocol,
    trials: int = 100,
    checkpoint_path: Path | str | None = None,
    # Root override
    root_dir: Path | str | None = None,
    # Optuna overrides
    study_name: str | None = None,
    # MLflow overrides — only meaningful when settings already has an [MLFLOW] section
    experiment_name: str | None = None,
    run_name: str | None = None,
    **additional_overrides: Any,
) -> OptimizationResult:
    """Run Optuna hyperparameter optimization with optional overrides.

    To enable MLflow tracking, include an ``[MLFLOW]`` section in your configuration.
    There is no boolean ``mlflow`` toggle.

    Args:
        settings: Parsed and validated configuration (with ``[OPTUNA].enabled=true``).
        trials: Number of trials to run.
        checkpoint_path: Optional warm-start checkpoint path.
        root_dir: Override the root directory for all path resolution.
        study_name: Override ``[OPTUNA].study_name``.
        experiment_name: Override ``[MLFLOW].experiment_name``.
        run_name: Override ``[MLFLOW].run_name``.
        additional_overrides: Extra overrides passed through to the manager.

    Returns:
        OptimizationResult: Optimization summary, best trial, and final training result.

    Raises:
        WorkflowError: On execution failure.
    """
    input_data = OptimizationCommandInput(
        trials=trials,
        checkpoint_path=checkpoint_path,
        root_dir=root_dir,
        study_name=study_name,
        experiment_name=experiment_name,
        run_name=run_name,
        additional_overrides=additional_overrides,
    )

    return _dispatcher.execute("optimize", input_data, settings)
