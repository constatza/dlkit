"""Settings package test fixtures."""

from __future__ import annotations

from typing import Any

import pytest


class MockTrial:
    """Mock Optuna trial for testing hyperparameter functionality."""

    def __init__(self):
        self.suggestions = {}

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        """Suggest integer value (returns midpoint for deterministic testing)."""
        value = (low + high) // 2
        self.suggestions[name] = value
        return value

    def suggest_float(
        self, name: str, low: float, high: float, step: float | None = None, log: bool = False
    ) -> float:
        """Suggest float value (returns midpoint for deterministic testing)."""
        value = (low + high) / 2
        self.suggestions[name] = value
        return value

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        """Suggest categorical value (returns first choice for deterministic testing)."""
        value = choices[0] if choices else None
        self.suggestions[name] = value
        return value


@pytest.fixture
def mock_trial() -> MockTrial:
    """Mock Optuna trial fixture for hyperparameter testing.

    Returns:
        MockTrial: Mock trial instance for testing
    """
    return MockTrial()


@pytest.fixture
def sample_build_context() -> Any:
    """Sample build context for integration testing.

    Returns:
        BuildContext: Sample build context instance
    """
    from pathlib import Path

    from dlkit.infrastructure.config.core.context import BuildContext

    return BuildContext(
        mode="training", device="cpu", random_seed=42, working_directory=Path.cwd(), overrides={}
    )


@pytest.fixture
def sample_general_settings_data() -> dict[str, Any]:
    """Sample data for TrainingJobConfig testing.

    Returns:
        Dict[str, Any]: Complete job configuration using new lowercase-section structure.
    """
    return {
        "run": {"type": "train", "seed": 1, "precision": "32"},
        "experiment": {"name": "test-experiment"},
        "model": {
            "class": "ConstantWidthFFNN",
            "module_path": "dlkit.domain.nn",
            "hidden_size": 64,
        },
        "data": {
            "class": "FlexibleDataset",
            "root": "/tmp/data",
            "batch_size": 32,
            "features": [{"name": "x", "path": "X.npy", "model_input": True}],
            "targets": [{"name": "y", "path": "y.npy"}],
        },
        "training": {
            "loss": "mse",
            "stopping": {"monitor": "val/loss", "patience": 10, "direction": "min"},
            "trainer": {"max_epochs": 100, "accelerator": "cpu"},
            "optimizer": {"name": "AdamW", "lr": 1e-3},
        },
    }


@pytest.fixture
def inference_config_data(tmp_path) -> dict[str, Any]:
    """Configuration data for inference mode testing.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        Dict[str, Any]: Inference mode configuration using new lowercase-section structure.
    """
    # Create a temporary checkpoint file for validation
    checkpoint_file = tmp_path / "model.ckpt"
    checkpoint_file.write_text("fake checkpoint")

    return {
        "run": {"type": "predict", "seed": 123},
        "model": {
            "class": "InferenceModel",
            "module_path": "dlkit.domain.nn.ffnn",
            "checkpoint": str(checkpoint_file),
        },
    }


@pytest.fixture
def invalid_inference_config_data() -> dict[str, Any]:
    """Invalid configuration data for inference mode (missing checkpoint).

    Returns:
        Dict[str, Any]: Invalid inference configuration using new lowercase-section structure.
    """
    return {
        "run": {"type": "predict", "seed": 42},
        "model": {
            "class": "InferenceModel",
            "module_path": "dlkit.domain.nn.ffnn",
            # Missing required checkpoint for inference
        },
    }


@pytest.fixture
def sample_toml_config_advanced() -> str:
    """Advanced TOML configuration for testing.

    Returns:
        str: TOML configuration content with complex structure using new lowercase-section format.
    """
    return """
[run]
type = "train"
seed = 999
precision = "32"

[experiment]
name = "advanced_experiment"

[model]
class = "AdvancedModel"
module_path = "dlkit.domain.nn.ffnn"
heads = 8
num_layers = 6
latent_size = 256
in_channels = 3
out_channels = 10

[data]
class = "AdvancedDataset"
module_path = "dlkit.engine.data.datasets"
batch_size = 128
num_workers = 8
shuffle = true
pin_memory = true

[data.splits]
train = 0.7
val = 0.2
test = 0.1

[training]
loss = "mse"

[training.trainer]
accelerator = "gpu"
devices = 2
strategy = "ddp"
max_epochs = 100

[training.optimizer]
name = "AdamW"
lr = 0.0001
weight_decay = 0.01

[training.scheduler]
name = "CosineAnnealingLR"
T_max = 100

[tracking]
backend = "mlflow"

[tracking.mlflow]
experiment_name = "advanced_experiment"
run_name = "advanced_run_001"
"""


@pytest.fixture
def malformed_toml_config() -> str:
    """Malformed TOML configuration for error testing.

    Returns:
        str: Invalid TOML content
    """
    return """
[run
type = "train"
# Missing closing bracket above

[model]
name = 
module_path = "test"
# Missing value for name
"""


@pytest.fixture
def optuna_model_config() -> str:
    """TOML configuration with Optuna/search job settings.

    Returns:
        str: TOML with search job configuration using new lowercase-section format.
    """
    return """
[run]
type = "search"
seed = 42

[experiment]
name = "optuna_experiment"

[model]
class = "OptunaModel"
module_path = "dlkit.domain.nn.ffnn"

[search]
n_trials = 10
direction = "minimize"
"""
