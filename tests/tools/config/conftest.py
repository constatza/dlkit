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
        self, name: str, low: float, high: float, step: float = None, log: bool = False
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

    from dlkit.tools.config.core.context import BuildContext

    return BuildContext(
        mode="training", device="cpu", random_seed=42, working_directory=Path.cwd(), overrides={}
    )


@pytest.fixture
def sample_general_settings_data() -> dict[str, Any]:
    """Sample dataflow for GeneralSettings testing.

    Returns:
        Dict[str, Any]: Complete general settings configuration
    """
    return {
        "SESSION": {
            "name": "test_general_session",
            "inference": False,  # Training mode
            "seed": 42,
            "precision": "medium",
        },
        "MODEL": {
            "name": "TestModel",
            "module_path": "test.models",
            "heads": 4,
            "num_layers": 3,
            "latent_size": 128,
        },
        "MLFLOW": {
            "enabled": True,
            "experiment_name": "test_general_experiment",
        },
        "OPTUNA": {"enabled": True, "n_trials": 50},
        "DATAMODULE": {
            "name": "TestDataModule",
            "module_path": "test.datamodules",
            "dataloader": {
                "batch_size": 64,
            },
        },
        "DATASET": {"name": "TestDataset", "module_path": "test.datasets"},
        "TRAINING": {
            "epochs": 20,
            "trainer": {"accelerator": "cpu", "devices": 1},
            "optimizer": {"name": "Adam", "lr": 0.001},
        },
    }


@pytest.fixture
def inference_config_data(tmp_path) -> dict[str, Any]:
    """Configuration dataflow for inference mode testing.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        Dict[str, Any]: Inference mode configuration
    """
    # Create a temporary checkpoint file for validation
    checkpoint_file = tmp_path / "model.ckpt"
    checkpoint_file.write_text("fake checkpoint")

    return {
        "SESSION": {
            "name": "inference_session",
            "inference": True,  # Inference mode
            "seed": 123,
            "precision": "medium",
        },
        "MODEL": {
            "name": "InferenceModel",
            "module_path": "test.models",
            "checkpoint": str(checkpoint_file),
        },
    }


@pytest.fixture
def invalid_inference_config_data() -> dict[str, Any]:
    """Invalid configuration dataflow for inference mode (missing checkpoint).

    Returns:
        Dict[str, Any]: Invalid inference configuration
    """
    return {
        "SESSION": {
            "name": "invalid_inference",
            "inference": True,  # Inference mode
            "seed": 42,
            "precision": "medium",
        },
        "MODEL": {
            "name": "InferenceModel",
            "module_path": "test.models",
            # Missing required checkpoint for inference
        },
    }


@pytest.fixture
def sample_toml_config_advanced() -> str:
    """Advanced TOML configuration for testing.

    Returns:
        str: TOML configuration content with complex structure
    """
    return """
[SESSION]
name = "advanced_session"
inference = false
seed = 999
precision = "high"

[MODEL]
name = "AdvancedModel"
module_path = "advanced.models"
heads = 8
num_layers = 6
latent_size = 256
in_channels = 3
out_channels = 10

[MLFLOW]
enabled = true
experiment_name = "advanced_experiment"
run_name = "advanced_run_001"
register_model = true

[OPTUNA]
enabled = true
n_trials = 200
direction = "minimize"

[OPTUNA.sampler]
name = "TPESampler"
seed = 42

[DATAMODULE]
name = "AdvancedDataModule"
module_path = "advanced.datamodules"
batch_size = 128
num_workers = 8

[DATAMODULE.dataloader]
shuffle = true
pin_memory = true
num_workers = 8

[DATASET]
name = "AdvancedDataset"
module_path = "advanced.datasets"
train_split = 0.7
val_split = 0.2
test_split = 0.1

[TRAINING]
epochs = 100
# gradient_clip_val should be in trainer section

[TRAINING.trainer]
accelerator = "gpu"
devices = 2
strategy = "ddp"
precision = "16-mixed"

[TRAINING.optimizer]
name = "AdamW"
lr = 0.0001
weight_decay = 0.01

[TRAINING.scheduler]
name = "CosineAnnealingLR"
T_max = 100

[PATHS]
output_dir = "output/advanced"
checkpoints_dir = "checkpoints/advanced"
figures_dir = "figures/advanced"
"""


@pytest.fixture
def malformed_toml_config() -> str:
    """Malformed TOML configuration for error testing.

    Returns:
        str: Invalid TOML content
    """
    return """
[SESSION
name = "malformed"
inference = false
# Missing closing bracket above

[MODEL]
name = 
module_path = "test"
# Missing value for name
"""


@pytest.fixture
def optuna_model_config() -> str:
    """TOML configuration with Optuna model settings.

    Returns:
        str: TOML with Optuna model configuration
    """
    return """
[SESSION]
name = "optuna_session"
inference = false

[MODEL]
name = "OptunaModel"
module_path = "test.models"

[OPTUNA]
enabled = true
n_trials = 10

# OPTUNA model_params would be configured differently in the new architecture
# This test may need to be updated for the new hyperparameter handling
"""
