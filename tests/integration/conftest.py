"""Integration test fixtures using direct JobConfig objects (no TOML).

These fixtures generate tiny synthetic datasets and construct
typed workflow config instances programmatically via model_validate to avoid
relying on TOML parsing. This keeps integration tests fast and robust.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from dlkit.common import TrainingResult
from dlkit.infrastructure.config.job_config import (
    TrainingJobConfig,
)

# Test constants - optimized for speed
FEATURE_SIZE: int = 4
TARGET_SIZE: int = 2
NUM_SAMPLES: int = 20  # Reduced from 100 for faster testing
BATCH_SIZE: int = 4  # Reduced from 8 for faster testing
EPOCHS: int = 1  # Reduced from 2 for faster testing
OPTUNA_TRIALS: int = 2

# Graph-specific constants
NUM_NODES: int = 5  # Small graph for fast testing
NODE_FEATURES: int = 3  # Node feature dimension
EDGE_FEATURES: int = 2  # Edge feature dimension
NUM_GRAPHS: int = 10  # Number of graphs in dataset


@pytest.fixture
def minimal_dataset(tmp_path: Path) -> dict[str, Path]:
    """Create minimal supervised learning dataset for fast testing.

    Creates very small synthetic dataflow files suitable for testing workflows
    without requiring large amounts of compute time.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Dictionary of paths to created dataset files.
    """
    # Generate small synthetic dataflow
    np.random.seed(42)  # Reproducible test dataflow
    X = np.random.randn(NUM_SAMPLES, FEATURE_SIZE).astype(np.float32)
    y = np.random.randint(0, TARGET_SIZE, size=(NUM_SAMPLES, 1)).astype(np.float32)

    # Write dataflow files
    data_dir = tmp_path / "dataflow"
    data_dir.mkdir(parents=True, exist_ok=True)

    X_path = data_dir / "features.npy"
    y_path = data_dir / "targets.npy"

    np.save(X_path, X)
    np.save(y_path, y)

    return {
        "features": X_path,
        "targets": y_path,
        "data_dir": data_dir,
    }


@pytest.fixture
def minimal_graph_dataset(tmp_path: Path) -> dict[str, Path]:
    """Create minimal graph dataset for fast testing.

    Creates small synthetic graph data suitable for testing graph workflows
    without requiring large amounts of compute time.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Dictionary of paths to created graph dataset files.
    """
    # Generate small synthetic graph data
    np.random.seed(42)  # Reproducible test data

    # Node features: (NUM_GRAPHS, NUM_NODES, NODE_FEATURES)
    x = np.random.randn(NUM_GRAPHS, NUM_NODES, NODE_FEATURES).astype(np.float32)

    # Adjacency matrix: (NUM_NODES, NUM_NODES) - same for all graphs
    adjacency = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    adjacency[0, 1] = 1.0
    adjacency[1, 2] = 1.0
    adjacency[2, 3] = 1.0
    adjacency[3, 4] = 1.0
    adjacency[4, 0] = 1.0

    # Target: (NUM_GRAPHS, NUM_NODES, TARGET_SIZE)
    y = np.random.randn(NUM_GRAPHS, NUM_NODES, TARGET_SIZE).astype(np.float32)

    # Write data files
    data_dir = tmp_path / "graph_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    x_path = data_dir / "node_features.npy"
    adjacency_path = data_dir / "adjacency.npy"
    y_path = data_dir / "targets.npy"

    np.save(x_path, x)
    np.save(adjacency_path, adjacency)
    np.save(y_path, y)

    return {
        "node_features": x_path,
        "adjacency": adjacency_path,
        "targets": y_path,
        "data_dir": data_dir,
    }


@pytest.fixture
def minimal_model_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal pre-trained model checkpoint for inference testing.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to the created checkpoint file.
    """
    checkpoint_path = tmp_path / "model.ckpt"

    # Build the model to get the correct state dict keys
    from dlkit.domain.nn.ffnn import FFNN

    _model = FFNN(
        in_features=FEATURE_SIZE,
        out_features=TARGET_SIZE,
        hidden_size=FEATURE_SIZE,
        num_layers=0,
    )
    checkpoint_payload = {
        "state_dict": {f"model.{k}": v for k, v in _model.state_dict().items()},
        "dlkit_metadata": {
            "model_family": "dlkit_nn",
            "wrapper_type": "StandardLightningWrapper",
            "input_shapes": {"x": [FEATURE_SIZE]},
            "output_shapes": {"y": [TARGET_SIZE]},
            "model_settings": {
                "name": "FFNN",
                "module_path": "dlkit.domain.nn",
                "hyper_kwargs": {
                    "hidden_size": FEATURE_SIZE,
                    "num_layers": 0,
                },
            },
            "entry_configs": [
                {"name": "x", "class_name": "Feature"},
                {"name": "y", "class_name": "Target"},
            ],
        },
    }

    torch.save(checkpoint_payload, checkpoint_path)
    return checkpoint_path


def _make_training_job_config(
    *,
    feature_path: Path,
    target_path: Path,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    checkpoint: Path | None = None,
    extra_training: dict[str, Any] | None = None,
    extra_run: dict[str, Any] | None = None,
    extra_experiment: dict[str, Any] | None = None,
    extra_tracking: dict[str, Any] | None = None,
) -> TrainingJobConfig:
    """Build a minimal TrainingJobConfig.

    Uses FlexibleDataset with x/y entries and a small FFNN model.

    Args:
        feature_path: Path to feature .npy file.
        target_path: Path to target .npy file.
        batch_size: Number of samples per batch.
        epochs: Maximum number of training epochs.
        checkpoint: Optional model checkpoint path.
        extra_training: Additional overrides for the training section.
        extra_run: Additional overrides for the run section.
        extra_experiment: Additional overrides for the experiment section.
        extra_tracking: Additional overrides for the tracking section.

    Returns:
        TrainingJobConfig with minimal settings for integration testing.
    """
    model_dict: dict[str, Any] = {
        "class": "FFNN",
        "module_path": "dlkit.domain.nn",
        "hidden_size": FEATURE_SIZE,
        "num_layers": 0,
    }
    if checkpoint is not None:
        model_dict["checkpoint"] = str(checkpoint)

    training_dict: dict[str, Any] = {
        "loss": "mse",
        "trainer": {
            "fast_dev_run": True,
            "enable_checkpointing": False,
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "max_epochs": epochs,
        },
        "optimizer": {"name": "AdamW", "lr": 1e-3},
        "metrics": [{"name": "MeanSquaredError", "module_path": "dlkit.domain.metrics"}],
    }
    if extra_training:
        training_dict.update(extra_training)

    run_dict: dict[str, Any] = {"type": "train", "seed": 42}
    if extra_run:
        run_dict.update(extra_run)

    experiment_dict: dict[str, Any] = {"name": "integration_test"}
    if extra_experiment:
        experiment_dict.update(extra_experiment)

    payload: dict[str, Any] = {
        "run": run_dict,
        "experiment": experiment_dict,
        "model": model_dict,
        "data": {
            "class": "FlexibleDataset",
            "module_path": "dlkit.engine.data.datasets",
            "batch_size": batch_size,
            "num_workers": 0,
            "shuffle": True,
            "pin_memory": False,
            "persistent_workers": False,
            "features": [{"name": "x", "path": str(feature_path), "format": "npy"}],
            "targets": [{"name": "y", "path": str(target_path), "format": "npy"}],
        },
        "training": training_dict,
    }
    if extra_tracking:
        payload["tracking"] = extra_tracking

    return TrainingJobConfig.model_validate(payload)


@pytest.fixture
def training_settings(minimal_dataset: dict[str, Path], tmp_path: Path) -> TrainingJobConfig:
    """Create TrainingJobConfig for vanilla training integration tests (no TOML).

    Args:
        minimal_dataset: Fixture providing dataset paths.
        tmp_path: Pytest temporary directory fixture.

    Returns:
        TrainingJobConfig with minimal settings for training integration tests.
    """
    return _make_training_job_config(
        feature_path=minimal_dataset["features"],
        target_path=minimal_dataset["targets"],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )


@pytest.fixture
def inference_settings(
    minimal_dataset: dict[str, Path], minimal_model_checkpoint: Path, tmp_path: Path
) -> TrainingJobConfig:
    """Create TrainingJobConfig for inference integration tests (no TOML).

    Args:
        minimal_dataset: Fixture providing dataset paths.
        minimal_model_checkpoint: Fixture providing a pre-trained checkpoint.
        tmp_path: Pytest temporary directory fixture.

    Returns:
        TrainingJobConfig configured with a checkpoint for inference tests.
    """
    return _make_training_job_config(
        feature_path=minimal_dataset["features"],
        target_path=minimal_dataset["targets"],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        checkpoint=minimal_model_checkpoint,
    )


@pytest.fixture
def graph_settings(minimal_graph_dataset: dict[str, Path], tmp_path: Path) -> TrainingJobConfig:
    """Create TrainingJobConfig for graph model training integration tests.

    Uses GraphDataset with node features, adjacency matrix, and targets.
    Model is a small GProjection graph neural network.

    Args:
        minimal_graph_dataset: Fixture providing graph dataset paths.
        tmp_path: Pytest temporary directory fixture.

    Returns:
        TrainingJobConfig configured for graph workflow testing.
    """
    return TrainingJobConfig.model_validate(
        {
            "run": {"type": "train", "seed": 42},
            "experiment": {"name": "graph_integration_test"},
            "model": {
                "class": "GProjection",
                "module_path": "dlkit.domain.nn.graph.projection_networks",
                "hidden_size": 4,
                "in_channels": NODE_FEATURES,
                "out_channels": TARGET_SIZE,
            },
            "data": {
                "class": "GraphDataset",
                "module_path": "dlkit.engine.data.datasets.graph",
                "batch_size": 2,
                "num_workers": 0,
                "shuffle": True,
                "pin_memory": False,
                "persistent_workers": False,
                "root": str(minimal_graph_dataset["data_dir"]),
                "features": [
                    {
                        "name": "x",
                        "path": str(minimal_graph_dataset["node_features"]),
                        "format": "npy",
                    },
                    {
                        "name": "edge_index",
                        "path": str(minimal_graph_dataset["adjacency"]),
                        "format": "npy",
                    },
                ],
                "targets": [
                    {
                        "name": "y",
                        "path": str(minimal_graph_dataset["targets"]),
                        "format": "npy",
                    }
                ],
            },
            "training": {
                "loss": "mse",
                "trainer": {
                    "fast_dev_run": True,
                    "enable_checkpointing": False,
                    "accelerator": "cpu",
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                    "max_epochs": 1,
                },
                "optimizer": {"name": "AdamW", "lr": 1e-3},
                "metrics": [{"name": "MeanSquaredError", "module_path": "dlkit.domain.metrics"}],
            },
        }
    )


@pytest.fixture
def mlflow_settings(minimal_dataset: dict[str, Path], tmp_path: Path) -> TrainingJobConfig:
    """Create TrainingJobConfig with MLflow enabled.

    Args:
        minimal_dataset: Fixture providing dataset paths.
        tmp_path: Pytest temporary directory fixture.
    Returns:
        TrainingJobConfig with MLflow tracking configured.
    """
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow_uri = f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}"

    return _make_training_job_config(
        feature_path=minimal_dataset["features"],
        target_path=minimal_dataset["targets"],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        extra_experiment={"name": "test_experiment"},
        extra_tracking={"backend": "mlflow", "uri": mlflow_uri},
    )


@pytest.fixture
def expected_training_metrics() -> dict[str, Any]:
    """Expected structure and types for training metrics.

    Returns:
        Dictionary defining expected training result metrics.
    """
    return {
        "required_keys": ["duration_seconds"],
        "optional_keys": ["train_loss", "val_loss", "mlflow_run_id", "optuna_best_value"],
        "numeric_keys": ["duration_seconds"],
    }


@pytest.fixture
def expected_inference_result() -> dict[str, Any]:
    """Expected structure for inference results.

    Returns:
        Dictionary defining expected inference result structure.
    """
    return {
        "required_keys": ["predictions", "duration_seconds"],
        "optional_keys": ["metrics"],
        "predictions_shape": (4, TARGET_SIZE),  # test split size, output dimension
    }


@pytest.fixture
def create_test_training_result():
    """Factory fixture for creating TrainingResult instances for testing.

    Returns:
        Function that creates TrainingResult with specified parameters.
    """

    def _create_result(
        duration: float = 10.0,
        metrics: dict[str, Any] | None = None,
        artifacts: dict[str, Path] | None = None,
        model_state: Any = None,
    ) -> TrainingResult:
        return TrainingResult(
            model_state=model_state,
            metrics=metrics or {"train_loss": 0.5, "val_loss": 0.3},
            artifacts=artifacts or {},
            duration_seconds=duration,
        )

    return _create_result


@pytest.fixture(autouse=True)
def cleanup_mlflow_state():
    """Clean up MLflow global state between tests to prevent interference.

    MLflow's autologging and global tracking URI state can persist between tests
    and cause issues when Optuna tests run after MLflow tests.
    """
    # Clean up before test
    try:
        import mlflow

        mlflow.pytorch.autolog(disable=True)
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass

    yield

    # Clean up after test
    try:
        import mlflow

        mlflow.pytorch.autolog(disable=True)
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass


@pytest.fixture
def double_precision_settings(
    training_settings: TrainingJobConfig,
) -> TrainingJobConfig:
    """Create TrainingJobConfig configured for double (float64) precision training.

    Args:
        training_settings: Base training settings fixture.

    Returns:
        TrainingJobConfig with run.precision set to FULL_64.
    """
    from dlkit.infrastructure.precision.strategy import PrecisionStrategy

    return training_settings.model_copy(
        update={
            "run": training_settings.run.model_copy(update={"precision": PrecisionStrategy.FULL_64})
        }
    )


@pytest.fixture
def integration_test_timeout() -> int:
    """Timeout for integration tests in seconds.

    Returns:
        Maximum time allowed for integration tests to complete.
    """
    return 10  # Reasonable timeout for tiny sqlite-backed integration runs
