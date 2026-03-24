"""Integration test fixtures using direct Settings objects (no TOML).

These fixtures generate tiny synthetic datasets and construct
`GeneralSettings` instances programmatically to avoid relying on
TOML parsing. This keeps integration tests fast and robust.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from dlkit.core.shape_specs import create_shape_spec, ModelFamily
from dlkit.tools.config import GeneralSettings
from dlkit.interfaces.api.domain import TrainingResult
from dlkit.tools.config import (
    SessionSettings,
    DataModuleSettings,
    DatasetSettings,
    TrainingSettings,
)
from dlkit.tools.config.dataloader_settings import DataloaderSettings
from dlkit.tools.config.trainer_settings import TrainerSettings
from dlkit.tools.config.components.model_components import (
    ModelComponentSettings,
    MetricComponentSettings,
)
from dlkit.tools.config.data_entries import Feature, Target
from dlkit.tools.config.dataset_settings import IndexSplitSettings


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

    # Create train/val/test split in the format expected by dlkit
    # Use simple text format with indices separated by newlines
    train_indices = "0 1 2 3 4 5 6 7 8 9 10 11"  # 12 samples for training
    val_indices = "12 13 14 15"  # 4 samples for validation
    test_indices = "16 17 18 19"  # 4 samples for testing

    split_content = f"{train_indices}\n{val_indices}\n{test_indices}\n"

    # Write dataflow files
    data_dir = tmp_path / "dataflow"
    data_dir.mkdir(parents=True, exist_ok=True)

    X_path = data_dir / "features.npy"
    y_path = data_dir / "targets.npy"
    split_path = data_dir / "split.txt"  # Use .txt format expected by dlkit

    np.save(X_path, X)
    np.save(y_path, y)
    split_path.write_text(split_content)

    return {
        "features": X_path,
        "targets": y_path,
        "split": split_path,
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
    # Create a simple connected graph
    adjacency = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    # Add edges: 0-1, 1-2, 2-3, 3-4, 4-0 (cycle)
    adjacency[0, 1] = 1.0
    adjacency[1, 2] = 1.0
    adjacency[2, 3] = 1.0
    adjacency[3, 4] = 1.0
    adjacency[4, 0] = 1.0

    # Target: (NUM_GRAPHS, NUM_NODES, TARGET_SIZE)
    y = np.random.randn(NUM_GRAPHS, NUM_NODES, TARGET_SIZE).astype(np.float32)

    # Create train/val/test split
    train_indices = "0 1 2 3 4 5"  # 6 graphs for training
    val_indices = "6 7"  # 2 graphs for validation
    test_indices = "8 9"  # 2 graphs for testing

    split_content = f"{train_indices}\n{val_indices}\n{test_indices}\n"

    # Write data files
    data_dir = tmp_path / "graph_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    x_path = data_dir / "node_features.npy"
    adjacency_path = data_dir / "adjacency.npy"
    y_path = data_dir / "targets.npy"
    split_path = data_dir / "split.txt"

    np.save(x_path, x)
    np.save(adjacency_path, adjacency)
    np.save(y_path, y)
    split_path.write_text(split_content)

    return {
        "node_features": x_path,
        "adjacency": adjacency_path,
        "targets": y_path,
        "split": split_path,
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
    from dlkit.core.models.nn.ffnn import ConstantWidthFFNN

    _model = ConstantWidthFFNN(
        in_features=FEATURE_SIZE,
        out_features=TARGET_SIZE,
        hidden_size=FEATURE_SIZE,
        num_layers=1,
    )
    checkpoint_payload = {
        "state_dict": {f"model.{k}": v for k, v in _model.state_dict().items()},
        "dlkit_metadata": {
            "model_family": "dlkit_nn",
            "wrapper_type": "StandardLightningWrapper",
            "shape_summary": {
                "in_shapes": [[FEATURE_SIZE]],
                "out_shapes": [[TARGET_SIZE]],
            },
            "model_settings": {
                "name": "ConstantWidthFFNN",
                "module_path": "dlkit.core.models.nn.ffnn.simple",
                "params": {
                    "hidden_size": FEATURE_SIZE,
                    "num_layers": 1,
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


def _make_settings(
    *,
    data_dir: Path,
    output_dir: Path,
    inference: bool = False,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    checkpoint: Path | None = None,
) -> GeneralSettings:
    """Build a minimal GeneralSettings for training/inference.

    Uses FlexibleDataset with X/y entries and a small FFNN model.
    """
    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DatasetSettings(
        name="FlexibleDataset",
        module_path="dlkit.core.datasets",
        root=data_dir,
        features=(Feature(name="x", path=data_dir / "features.npy"),),
        targets=(Target(name="y", path=data_dir / "targets.npy"),),
        split=IndexSplitSettings(),
    )

    datamodule = DataModuleSettings(
        name="InMemoryModule",
        module_path="dlkit.core.datamodules",
        dataloader=DataloaderSettings(
            num_workers=0,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            persistent_workers=False,
        ),
    )

    model = ModelComponentSettings(
        name="ConstantWidthFFNN",
        module_path="dlkit.core.models.nn.ffnn.simple",
        hidden_size=4,  # Reduced from 8 for faster testing
        num_layers=1,
        checkpoint=checkpoint if inference else None,
    )

    training = TrainingSettings(
        epochs=epochs,
        trainer=TrainerSettings(
            fast_dev_run=True,  # Use fast dev run for ultra-fast testing
            enable_checkpointing=False,
            accelerator="cpu",
            enable_progress_bar=False,  # Disable progress bar for faster tests
            enable_model_summary=False,  # Disable model summary for faster tests
        ),
        metrics=(
            MetricComponentSettings(
                name="MeanSquaredError",
                module_path="dlkit.core.training.metrics",
            ),
        ),
    )

    session = SessionSettings(name="integration_test", inference=inference, seed=42)

    return GeneralSettings(
        SESSION=session,
        MLFLOW=None,
        OPTUNA=None,
        DATAMODULE=datamodule,
        DATASET=dataset,
        TRAINING=training,
        MODEL=model,
    )


@pytest.fixture
def training_settings(minimal_dataset: dict[str, Path], tmp_path: Path) -> GeneralSettings:
    """Create GeneralSettings for vanilla training integration tests (no TOML)."""
    return _make_settings(
        data_dir=minimal_dataset["data_dir"],
        output_dir=tmp_path / "outputs",
        inference=False,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )


@pytest.fixture
def inference_settings(
    minimal_dataset: dict[str, Path], minimal_model_checkpoint: Path, tmp_path: Path
) -> GeneralSettings:
    """Create GeneralSettings for inference integration tests (no TOML)."""
    return _make_settings(
        data_dir=minimal_dataset["data_dir"],
        output_dir=tmp_path / "outputs",
        inference=True,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        checkpoint=minimal_model_checkpoint,
    )


@pytest.fixture
def graph_settings(minimal_graph_dataset: dict[str, Path], tmp_path: Path) -> GeneralSettings:
    """Create GeneralSettings for graph model training integration tests.

    Uses GraphDataset with node features, adjacency matrix, and targets.
    Model is a small GProjection graph neural network.

    Args:
        minimal_graph_dataset: Fixture providing graph dataset paths
        tmp_path: Pytest temporary directory fixture

    Returns:
        GeneralSettings configured for graph workflow testing
    """
    output_dir = tmp_path / "graph_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DatasetSettings(
        name="GraphDataset",
        module_path="dlkit.core.datasets.graph",
        root=minimal_graph_dataset["data_dir"],
        x=minimal_graph_dataset["node_features"],
        edge_index=minimal_graph_dataset["adjacency"],
        y=minimal_graph_dataset["targets"],
    )

    datamodule = DataModuleSettings(
        name="GraphDataModule",
        module_path="dlkit.core.datamodules.graph",
        dataloader=DataloaderSettings(
            num_workers=0,
            batch_size=2,  # Small batch for graph data
            shuffle=True,
            pin_memory=False,
            persistent_workers=False,
        ),
    )

    # Create explicit shape spec for graph model
    # Graph models need: x (node features), y (targets)
    shape_spec = create_shape_spec(
        shapes={
            "x": (NODE_FEATURES,),  # Node feature dimension (batch-free)
            "y": (TARGET_SIZE,),  # Output dimension (batch-free)
        },
        default_input="x",
        default_output="y",
        model_family=ModelFamily.GRAPH,  # Specify graph model family
    )

    model = ModelComponentSettings(
        name="GProjection",
        module_path="dlkit.core.models.nn.graph.projection_networks",
        hidden_size=4,  # Small hidden size for fast testing
        unified_shape=shape_spec,  # Provide explicit shape spec
    )

    training = TrainingSettings(
        epochs=1,
        trainer=TrainerSettings(
            fast_dev_run=True,  # Use fast dev run for ultra-fast testing
            enable_checkpointing=False,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
        ),
        metrics=(
            MetricComponentSettings(
                name="MeanSquaredError",
                module_path="dlkit.core.training.metrics",
            ),
        ),
    )

    session = SessionSettings(name="graph_integration_test", inference=False, seed=42)

    return GeneralSettings(
        SESSION=session,
        MLFLOW=None,
        OPTUNA=None,
        DATAMODULE=datamodule,
        DATASET=dataset,
        TRAINING=training,
        MODEL=model,
    )


@pytest.fixture
def mlflow_settings(
    minimal_dataset: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> GeneralSettings:
    """Create GeneralSettings with minimal MLflow-like training setup using overrides."""
    import dlkit.runtime.workflows.strategies.tracking.uri_resolver as uri_resolver

    base_settings = _make_settings(
        data_dir=minimal_dataset["data_dir"],
        output_dir=tmp_path / "outputs",
        inference=False,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    # Route select_backend() to a per-test isolated SQLite DB and suppress the
    # local-server probe. Setting MLFLOW_TRACKING_URI to a sqlite:/// URI is now
    # honoured by select_backend(), so both the env var and the probe must be set.
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}",
    )
    mlartifacts_dir = tmp_path / "mlartifacts"
    mlartifacts_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MLFLOW_ARTIFACT_URI", mlartifacts_dir.as_uri())
    monkeypatch.setattr(uri_resolver, "local_host_alive", lambda: False)

    from dlkit.tools.config.mlflow_settings import MLflowSettings

    mlflow_cfg = MLflowSettings(experiment_name="test_experiment")
    settings_with_mlflow = base_settings.model_copy(update={"MLFLOW": mlflow_cfg})

    return settings_with_mlflow


@pytest.fixture
def optuna_settings(minimal_dataset: dict[str, Path], tmp_path: Path) -> GeneralSettings:
    """Create GeneralSettings with Optuna enabled using overrides."""
    from dlkit.interfaces.api.overrides.manager import BasicOverrideManager

    # Start with base training settings
    base_settings = _make_settings(
        data_dir=minimal_dataset["data_dir"],
        output_dir=tmp_path / "outputs",
        inference=False,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    # Enable Optuna using overrides
    manager = BasicOverrideManager()
    optuna_settings = manager.apply_overrides(
        base_settings, enable_optuna=True, trials=OPTUNA_TRIALS, study_name="test_study"
    )
    # Ensure isolated study storage per test to avoid cross-test accumulation
    try:
        unique_storage = f"sqlite:///{(tmp_path / 'optuna.db').as_posix()}"
        new_optuna = optuna_settings.OPTUNA.model_copy(
            update={
                "storage": unique_storage,
                "study_name": f"test_study_{tmp_path.name}",
            }
        )
        return optuna_settings.model_copy(update={"OPTUNA": new_optuna})
    except Exception:
        return optuna_settings


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
def optuna_config_content(minimal_dataset: dict[str, Path], tmp_path: Path) -> str:
    """Produce a minimal TOML config string enabling Optuna.

    Uses the same tmp_path-backed dataset created by minimal_dataset so relative
    resolution works when the test writes this content to a file in tmp_path.
    """
    data_dir = minimal_dataset["data_dir"]
    # Keep paths relative to DATASET.root by specifying filenames only
    return f"""
[PATHS]
output_dir = "outputs"

[SESSION]
name = "integration_test"
inference = false
seed = 42

[DATASET]
name = "FlexibleDataset"
module_path = "dlkit.core.datasets"
root_dir = "{data_dir.as_posix()}"

[[DATASET.features]]
name = "x"
path = "features.npy"

[[DATASET.targets]]
name = "y"
path = "targets.npy"

[DATASET.split]
filepath = "split.txt"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
num_workers = 0
batch_size = {BATCH_SIZE}
shuffle = true
pin_memory = false
persistent_workers = false

[MODEL]
name = "ConstantWidthFFNN"
module_path = "dlkit.core.models.nn.ffnn.simple"
hidden_size = 4
num_layers = 1

[TRAINING]
epochs = {EPOCHS}

[TRAINING.trainer]
max_steps = 1
enable_checkpointing = false
accelerator = "cpu"

[OPTUNA]
enabled = true
n_trials = {OPTUNA_TRIALS}
direction = "minimize"
study_name = "test_study"
"""


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

        # Disable autologging
        mlflow.pytorch.autolog(disable=True)
        # End any active runs
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass

    yield

    # Clean up after test
    try:
        import mlflow

        # Disable autologging
        mlflow.pytorch.autolog(disable=True)
        # End any active runs
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass


@pytest.fixture
def double_precision_settings(training_settings: GeneralSettings) -> GeneralSettings:
    """Create GeneralSettings configured for double (float64) precision training.

    Patches the base training settings to use FULL_64 precision, exercising
    Lightning's DoublePrecisionPlugin which applies apply_to_collection on batches.

    Args:
        training_settings: Base training settings fixture.

    Returns:
        GeneralSettings with SESSION.precision set to FULL_64.
    """
    from dlkit.tools.config.precision import PrecisionStrategy

    new_session = training_settings.SESSION.model_copy(
        update={"precision": PrecisionStrategy.FULL_64}
    )
    return training_settings.model_copy(update={"SESSION": new_session})


@pytest.fixture
def integration_test_timeout() -> int:
    """Timeout for integration tests in seconds.

    Returns:
        Maximum time allowed for integration tests to complete.
    """
    return 10  # Reasonable timeout for tiny sqlite-backed integration runs
