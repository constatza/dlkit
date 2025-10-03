"""Type definitions for DLKit API overrides."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict


class BasicOverrides(TypedDict, total=False):
    """Type definition for basic runtime overrides.

    All fields are optional to allow partial overrides.
    """

    # Path overrides
    checkpoint_path: Path
    output_dir: Path
    data_dir: Path

    # Training overrides
    epochs: int
    batch_size: int
    learning_rate: float

    # MLflow server overrides
    mlflow_host: str
    mlflow_port: int

    # MLflow client overrides
    experiment_name: str
    run_name: str

    # Optuna overrides
    trials: int
    study_name: str


class MLflowOverrides(TypedDict, total=False):
    """Type definition for MLflow-specific overrides."""

    # Server configuration
    host: str
    port: int
    backend_store_uri: str
    artifacts_destination: str

    # Client configuration
    experiment_name: str
    run_name: str
    register_model: bool


class TrainingOverrides(TypedDict, total=False):
    """Type definition for training-specific overrides."""

    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    scheduler: str

    # Training flags
    train: bool
    test: bool
    predict: bool


class PathOverrides(TypedDict, total=False):
    """Type definition for path-related overrides."""

    root_dir: Path
    input_dir: Path
    output_dir: Path
    data_dir: Path
    checkpoints_dir: Path
    figures_dir: Path
    predictions_dir: Path
