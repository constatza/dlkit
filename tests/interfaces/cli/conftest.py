"""CLI test fixtures and utilities."""

from __future__ import annotations
import pytest
from typer.testing import CliRunner

import sys
from pathlib import Path
from unittest.mock import Mock
from typing import Any

# Disable Rich for testing to prevent colors and formatting
sys.modules.setdefault("rich", Mock())
sys.modules.setdefault("rich.console", Mock())


@pytest.fixture
def cli_runner() -> CliRunner:
    """Typer CLI test runner fixture with colors disabled."""
    return CliRunner(
        env={
            "NO_COLOR": "1",
            "CLICOLOR": "0",
            "FORCE_COLOR": "0",
            "PY_COLORS": "0",
            "RICH_FORCE_TERMINAL": "0",
            "TERM": "dumb",
        }
    )


@pytest.fixture
def sample_config_content(tmp_path: Path) -> str:
    """Sample TOML configuration content for CLI tests.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Valid TOML configuration string for testing.
    """

    def to_toml_path(path: Path) -> str:
        # Use POSIX-style paths to keep TOML strings portable across platforms
        return path.as_posix()

    root_dir = to_toml_path(tmp_path)
    feature_path = to_toml_path(tmp_path / "X.npy")
    target_path = to_toml_path(tmp_path / "Y.npy")
    split_path = to_toml_path(tmp_path / "indices.txt")
    checkpoint_path = to_toml_path(tmp_path / "model.ckpt")
    default_root = to_toml_path(tmp_path / "work")

    return f"""[SESSION]
name = "test_session"
inference = false
seed = 42

[DATASET]
name = "FlexibleDataset"
root_dir = "{root_dir}"

[[DATASET.features]]
name = "X"
path = "{feature_path}"

[[DATASET.targets]]
name = "Y"
path = "{target_path}"

[DATASET.split]
filepath = "{split_path}"

[MODEL]
name = "ConstantWidthFFNN"
module_path = "dlkit.core.models.nn.ffnn.simple"
checkpoint = "{checkpoint_path}"

[MLFLOW]
enabled = true

[MLFLOW.client]
experiment_name = "test_experiment"

[TRAINING.trainer]
default_root_dir = "{default_root}"
"""


@pytest.fixture
def sample_config_file(tmp_path: Path, sample_config_content: str) -> Path:
    """Create a sample configuration file for CLI testing.

    Args:
        tmp_path: Pytest temporary directory fixture.
        sample_config_content: Configuration content fixture.

    Returns:
        Path to the created configuration file.
    """
    config_file = tmp_path / "test_config.toml"
    config_file.write_text(sample_config_content)

    # Create required files referenced in config
    (tmp_path / "X.npy").write_text("dummy")
    (tmp_path / "Y.npy").write_text("dummy")
    (tmp_path / "indices.txt").write_text("dummy")
    (tmp_path / "model.ckpt").write_text("dummy")
    (tmp_path / "work").mkdir(exist_ok=True)

    return config_file


@pytest.fixture
def mock_dependencies() -> dict[str, Mock]:
    """Mock external dependencies for CLI testing.

    Returns:
        Dictionary of mocked dependencies with autospec.
    """
    # Mock system dependencies that info command checks
    mock_torch = Mock()
    mock_torch.__version__ = "2.4.0"

    mock_lightning = Mock()
    mock_lightning.__version__ = "2.3.2"

    mock_mlflow = Mock()
    mock_mlflow.version = {"VERSION": "3.0.0"}

    mock_optuna = Mock()
    mock_optuna.__version__ = "3.7.0"

    return {
        "torch": mock_torch,
        "lightning": mock_lightning,
        "mlflow": mock_mlflow,
        "optuna": mock_optuna,
    }


@pytest.fixture
def mock_api_functions() -> dict[str, Mock]:
    """Mock API functions used by CLI commands.

    Returns:
        Dictionary of mocked API functions with proper return values.
    """
    from dlkit.interfaces.api.domain import ConfigurationError, TrainingResult

    # Mock successful validation
    mock_validate = Mock(return_value=True)

    # Mock successful training result
    from dlkit.interfaces.api.domain import ModelState

    mock_model_state = Mock(spec=ModelState)

    mock_training_result = TrainingResult(
        model_state=mock_model_state,
        duration_seconds=120.5,
        metrics={"train_loss": 0.25, "val_accuracy": 0.95},
        artifacts={"model": Path("model.ckpt"), "logs": Path("training.log")},
    )

    # Create successful training return
    mock_train = Mock(return_value=mock_training_result)

    # Mock successful config loading
    mock_settings = Mock()
    mock_settings.MLFLOW = Mock(is_active=False)
    mock_settings.OPTUNA = Mock(is_active=False)

    mock_load_config = Mock(return_value=mock_settings)

    # Mock failed config loading
    mock_config_error = ConfigurationError("Invalid configuration", {"config_path": "invalid.toml"})
    mock_failed_load_config = Mock(side_effect=mock_config_error)

    # Mock failed validation
    mock_failed_validate = Mock(side_effect=mock_config_error)

    # Create failed training mock that raises exception
    def mock_failed_train_side_effect(*args, **kwargs):
        raise mock_config_error

    mock_failed_train = Mock(side_effect=mock_failed_train_side_effect)

    return {
        "validate_config": mock_validate,
        "api_train": mock_train,
        "load_config": mock_load_config,
        "settings": mock_settings,
        "failed_load_config": mock_failed_load_config,
        "failed_validate": mock_failed_validate,
        "failed_train": mock_failed_train,
        "training_result": mock_training_result,
        "config_error": mock_config_error,
    }


@pytest.fixture
def cli_mock_patches() -> dict[str, str]:
    """Mapping of CLI modules to their mock patches.

    Returns:
        Dictionary mapping module names to patch paths.
    """
    return {
        "config_adapter": "dlkit.interfaces.cli.adapters.config_adapter",
        "error_handler": "dlkit.interfaces.cli.middleware.error_handler",
        "result_presenter": "dlkit.interfaces.cli.adapters.result_presenter",
        "api_train": "dlkit.interfaces.cli.commands.train.api_train",
        "api_validate": "dlkit.interfaces.cli.commands.train.validate_config",
        "api_infer": "dlkit.interfaces.api.infer",
        "api_optimize": "dlkit.interfaces.api.optimize",
    }


@pytest.fixture
def cli_exit_codes() -> dict[str, int]:
    """Standard CLI exit codes for testing.

    Returns:
        Dictionary mapping exit scenarios to expected codes.
    """
    return {
        "success": 0,
        "error": 1,
        "keyboard_interrupt": 1,
    }


@pytest.fixture
def expected_help_patterns() -> dict[str, str]:
    """Expected patterns in CLI help output.

    Returns:
        Dictionary of help text patterns for validation.
    """
    return {
        "main_help": "Deep Learning Toolkit",
        "usage": "Usage:",
        "commands": "Commands:",
        "train_help": "Training",
        "infer_help": "Inference",
        "optimize_help": "Hyperparameter optimization",
        "config_help": "Configuration validation",
        "server_help": "Server management",
    }


@pytest.fixture
def version_info_patterns() -> dict[str, str]:
    """Expected patterns in version and info output.

    Returns:
        Dictionary of version/info output patterns.
    """
    return {
        "version_title": "Version",
        "version_text": "DLKit v",
        "info_title": "System Information",
        "dependencies": "Dependencies:",
        "pytorch": "PyTorch:",
        "lightning": "Lightning:",
        "python": "Python:",
    }


@pytest.fixture
def sample_config_path(tmp_path: Path, sample_config_content: str) -> Path:
    """Create a sample configuration file path for testing.

    Args:
        tmp_path: Pytest temporary directory fixture.
        sample_config_content: Configuration content fixture.

    Returns:
        Path to the created configuration file.
    """
    config_file = tmp_path / "test_config.toml"
    config_file.write_text(sample_config_content)

    # Create required files referenced in config
    (tmp_path / "X.npy").write_text("dummy")
    (tmp_path / "Y.npy").write_text("dummy")
    (tmp_path / "indices.txt").write_text("dummy")
    (tmp_path / "model.ckpt").write_text("dummy")
    (tmp_path / "work").mkdir(exist_ok=True)

    return config_file


@pytest.fixture
def sample_checkpoint_path(tmp_path: Path) -> Path:
    """Create a sample checkpoint file path for testing.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to the created checkpoint file.
    """
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_text("dummy checkpoint content")
    return checkpoint_path


@pytest.fixture
def sample_settings() -> Mock:
    """Create mock settings object for testing.

    Returns:
        Mock GeneralSettings object configured for testing.
    """
    mock_settings = Mock()
    mock_mlflow = Mock()
    mock_mlflow.is_active = True
    mock_optuna = Mock()
    mock_optuna.is_active = False

    mock_settings.MLFLOW = mock_mlflow
    mock_settings.OPTUNA = mock_optuna
    mock_settings._inner_value = mock_settings

    # Mock model_dump to return a proper dict structure
    mock_settings.model_dump.return_value = {
        "SESSION": {"name": "test_session", "inference": False, "seed": 42},
        "DATASET": {"name": "FlexibleDataset", "root_dir": "."},
        "MODEL": {"name": "ConstantWidthFFNN", "module_path": "dlkit.core.models.nn.ffnn.simple"},
        "MLFLOW": {"enabled": True, "client": {"experiment_name": "test_experiment"}},
    }
    return mock_settings


# Note: A single inference-result fixture is provided below that returns an
# InferenceResult-like object. Avoid duplicate fixtures with the same name.


@pytest.fixture
def mock_configuration_error() -> Any:
    """Create mock configuration error for testing.

    Returns:
        Mock ConfigurationError object.
    """
    from dlkit.interfaces.api.domain import ConfigurationError

    return ConfigurationError(
        "Invalid configuration for testing",
        {"config_path": "test_config.toml", "field": "invalid_field"},
    )


@pytest.fixture
def mock_workflow_error() -> Any:
    """Create mock workflow error for testing.

    Returns:
        Mock WorkflowError object.
    """
    from dlkit.interfaces.api.domain import WorkflowError

    return WorkflowError(
        "Workflow execution failed for testing",
        {"stage": "inference", "reason": "model_load_failed"},
    )


@pytest.fixture
def create_success_result():
    """Factory fixture to create successful result values.

    Returns:
        Function that returns the value directly.
    """

    def _create_success(value: Any, command_name: str = "test_command") -> Any:
        return value

    return _create_success


@pytest.fixture
def create_failure_side_effect():
    """Factory fixture to create failing side effects.

    Returns:
        Function that creates side effect that raises the error.
    """

    def _create_failure(error: Any, command_name: str = "test_command") -> Any:
        def side_effect(*args, **kwargs):
            raise error

        return side_effect

    return _create_failure


@pytest.fixture
def mock_console() -> Mock:
    """Create mock console for result presentation testing.

    Returns:
        Mock Rich Console object for CLI output testing.
    """
    mock_console = Mock()
    mock_console.print = Mock()
    return mock_console


@pytest.fixture
def mock_successful_training_result() -> Mock:
    """Create mock successful training result.

    Returns:
        Mock TrainingResult object for testing.
    """
    from dlkit.interfaces.api.domain import TrainingResult, ModelState
    from pathlib import Path

    # Create a mock ModelState
    mock_model_state = Mock(spec=ModelState)

    return TrainingResult(
        model_state=mock_model_state,
        duration_seconds=10000.5,
        metrics={"train_loss": 0.15, "val_accuracy": 0.92},
        artifacts={"model": Path("model.ckpt"), "logs": Path("train.log")},
    )


@pytest.fixture
def mock_successful_inference_result() -> Mock:
    """Create mock successful inference result.

    Returns:
        Mock InferenceResult object for testing.
    """
    from dlkit.interfaces.api.domain import InferenceResult, ModelState

    # Create a mock ModelState
    mock_model_state = Mock(spec=ModelState)

    return InferenceResult(
        model_state=mock_model_state,
        predictions=[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]],
        metrics={"accuracy": 0.95, "loss": 0.05},
        duration_seconds=5.5,
    )


@pytest.fixture
def mock_strategy_error() -> Any:
    """Create mock strategy error for testing.

    Returns:
        Mock StrategyError object.
    """
    from dlkit.interfaces.api.domain import StrategyError

    return StrategyError(
        "Invalid strategy configuration for testing",
        {"strategy": "invalid_strategy", "field": "invalid_field"},
    )


@pytest.fixture
def train_config_templates() -> dict[str, str]:
    """Templates for different training configuration scenarios.

    Returns:
        Dictionary of configuration templates for different scenarios.
    """
    return {
        "vanilla": """[SESSION]
name = "vanilla_train"
inference = false
seed = 42

[PATHS]
output_dir = "{output_dir}"

[TRAINING.trainer]
max_epochs = 10
""",
        "mlflow": """[SESSION]
name = "mlflow_train"
inference = false
seed = 42

[PATHS]
output_dir = "{output_dir}"

[MLFLOW]
enabled = true
is_active = true

[MLFLOW.client]
experiment_name = "test_experiment"
run_name = "test_run"

[TRAINING.trainer]
max_epochs = 10
""",
        "optuna": """[SESSION]
name = "optuna_train"
inference = false
seed = 42

[PATHS]
output_dir = "{output_dir}"

[OPTUNA]
enabled = true
is_active = true
n_trials = 20
study_name = "test_study"

[TRAINING.trainer]
max_epochs = 5
""",
        "both": """[SESSION]
name = "both_train"
inference = false
seed = 42

[PATHS]
output_dir = "{output_dir}"

[MLFLOW]
enabled = true
is_active = true

[MLFLOW.client]
experiment_name = "test_experiment"
run_name = "test_run"

[OPTUNA]
enabled = true
is_active = true
n_trials = 10
study_name = "test_study"

[TRAINING.trainer]
max_epochs = 5
""",
    }


@pytest.fixture
def train_override_scenarios() -> dict[str, dict[str, Any]]:
    """Test scenarios for various CLI parameter overrides.

    Returns:
        Dictionary mapping scenario names to override parameter dictionaries.
    """
    return {
        "basic_overrides": {
            "epochs": 25,
            "batch_size": 64,
            "learning_rate": 0.001,
            "output_dir": None,  # Will be set to tmp_path/custom_output
            "data_dir": None,  # Will be set to tmp_path/custom_data
        },
        "mlflow_overrides": {
            "strategy": "mlflow",
            "mlflow_host": "localhost",
            "mlflow_port": 5000,
            "experiment_name": "custom_experiment",
            "run_name": "custom_run",
        },
        "optuna_overrides": {
            "strategy": "optuna",
            "trials": 100,
            "study_name": "custom_study",
        },
        "checkpoint_resume": {
            "checkpoint": None,  # Will be set to tmp_path/model.ckpt
            "epochs": 50,
        },
        "validate_only": {
            "validate_only": True,
        },
    }


@pytest.fixture
def mock_settings_factory():
    """Factory function to create mock settings for different scenarios.

    Returns:
        Function that creates mock settings based on scenario.
    """

    def _create_mock_settings(
        scenario: str = "vanilla", *, mlflow_active: bool = False, optuna_active: bool = False
    ) -> Mock:
        mock_settings = Mock()

        # Create MLFLOW mock
        mock_mlflow = Mock()
        mock_mlflow.is_active = mlflow_active
        mock_mlflow.enabled = mlflow_active
        mock_settings.MLFLOW = mock_mlflow

        # Create OPTUNA mock
        mock_optuna = Mock()
        mock_optuna.is_active = optuna_active
        mock_optuna.enabled = optuna_active
        mock_settings.OPTUNA = mock_optuna

        return mock_settings

    return _create_mock_settings


@pytest.fixture
def sample_convert_result() -> Mock:
    """Create mock ConvertResult for CLI testing.

    Returns:
        Mock ConvertResult object with proper structure.
    """
    from dlkit.interfaces.api.commands.convert_command import ConvertResult
    from pathlib import Path

    return ConvertResult(output_path=Path("test_model.onnx"), opset=17, inputs=[(1, 3, 224, 224)])


@pytest.fixture
def mock_convert_command() -> Mock:
    """Create mock ConvertCommand for CLI testing.

    Returns:
        Mock ConvertCommand with autospec for proper method signatures.
    """
    from dlkit.interfaces.api.commands.convert_command import ConvertCommand

    mock_command = Mock(spec=ConvertCommand)
    mock_command.execute = Mock()
    return mock_command


@pytest.fixture
def valid_shape_strings() -> list[str]:
    """Valid shape string examples for testing.

    Returns:
        List of valid shape string formats.
    """
    return [
        "3,224,224",  # Standard image format
        "784",  # Flattened vector
        "3,32,32",  # Smaller image
        "1,256",  # Sequence dataflow
        "3,224,224;10",  # Multiple inputs
        "3x224x224",  # Alternative separator
    ]


@pytest.fixture
def invalid_shape_strings() -> list[str]:
    """Invalid shape string examples for testing.

    Returns:
        List of invalid shape string formats.
    """
    return [
        "",  # Empty string
        "0,224,224",  # Zero dimension
        "-1,224,224",  # Negative dimension
        "abc,224,224",  # Non-numeric
        "3,224,",  # Trailing comma
        ",224,224",  # Leading comma
    ]


@pytest.fixture
def convert_cli_scenarios() -> dict[str, dict[str, Any]]:
    """CLI parameter scenarios for convert command testing.

    Returns:
        Dictionary mapping scenario names to parameter combinations.
    """
    return {
        "shape_basic": {
            "args": ["checkpoint.ckpt", "output.onnx", "--shape", "3,224,224"],
            "expected_shape": "3,224,224",
            "expected_batch_size": 1,
            "expected_opset": 17,
        },
        "shape_with_batch": {
            "args": ["checkpoint.ckpt", "output.onnx", "--shape", "3,224,224", "--batch-size", "4"],
            "expected_shape": "3,224,224",
            "expected_batch_size": 4,
            "expected_opset": 17,
        },
        "config_basic": {
            "args": ["checkpoint.ckpt", "output.onnx", "--config", "config.toml"],
            "expected_shape": None,
            "expected_batch_size": 1,
            "expected_opset": 17,
        },
        "config_with_batch": {
            "args": [
                "checkpoint.ckpt",
                "output.onnx",
                "--config",
                "config.toml",
                "--batch-size",
                "8",
            ],
            "expected_shape": None,
            "expected_batch_size": 8,
            "expected_opset": 17,
        },
        "custom_opset": {
            "args": ["checkpoint.ckpt", "output.onnx", "--shape", "784", "--opset", "11"],
            "expected_shape": "784",
            "expected_batch_size": 1,
            "expected_opset": 11,
        },
        "multiple_inputs": {
            "args": ["checkpoint.ckpt", "output.onnx", "--shape", "3,224,224;10"],
            "expected_shape": "3,224,224;10",
            "expected_batch_size": 1,
            "expected_opset": 17,
        },
    }


@pytest.fixture
def convert_error_scenarios() -> dict[str, dict[str, Any]]:
    """Error scenarios for convert command testing.

    Returns:
        Dictionary mapping error scenario names to test parameters.
    """
    return {
        "missing_both_params": {
            "args": ["checkpoint.ckpt", "output.onnx"],
            "expected_error": "Provide either --shape",
            "exit_code": 1,
        },
        "nonexistent_checkpoint": {
            "args": ["nonexistent.ckpt", "output.onnx", "--shape", "3,224,224"],
            "expected_error": "Export failed:",
            "exit_code": 1,
        },
        "invalid_opset": {
            "args": ["checkpoint.ckpt", "output.onnx", "--shape", "3,224,224", "--opset", "5"],
            "expected_error": "Export failed:",
            "exit_code": 1,
        },
    }
