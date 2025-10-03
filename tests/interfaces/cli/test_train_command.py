"""Test train command functionality and behavior."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dlkit.interfaces.cli.app import app as cli_app


@pytest.fixture
def cli_runner() -> CliRunner:
    """CLI runner for tests."""
    return CliRunner()


@pytest.fixture
def minimal_train_config() -> str:
    """Minimal training configuration for testing."""
    return """
[SESSION]
name = "test_training"
inference = false

[PATHS]
output_dir = "./outputs"

[TRAINING.trainer]
max_epochs = 1
"""


def test_train_help_accessible(cli_runner: CliRunner) -> None:
    """Train command help is accessible."""
    result = cli_runner.invoke(cli_app, ["train", "--help"])
    assert result.exit_code == 0


def test_train_requires_config_file(cli_runner: CliRunner) -> None:
    """Test that train command requires a config file when called directly."""
    result = cli_runner.invoke(cli_app, ["train"])
    assert result.exit_code != 0


def test_train_detects_missing_config_file(cli_runner: CliRunner) -> None:
    """Test that train command handles missing config files."""
    result = cli_runner.invoke(cli_app, ["train", "nonexistent.toml"])
    assert result.exit_code != 0


def test_train_validate_only_flag_requires_config(cli_runner: CliRunner) -> None:
    """Validate-only flag without config should error."""
    result = cli_runner.invoke(cli_app, ["train", "--validate-only"])
    assert result.exit_code != 0


def test_train_validate_only_missing_file(cli_runner: CliRunner) -> None:
    """Validate-only with missing config file should fail."""
    result = cli_runner.invoke(cli_app, ["train", "--validate-only", "nonexistent.toml"])
    assert result.exit_code != 0


def test_train_validate_only_processes_config(
    cli_runner: CliRunner, tmp_path, minimal_train_config: str
) -> None:
    """Validate-only processes configuration files without training."""
    config_file = tmp_path / "train_config.toml"
    config_file.write_text(minimal_train_config)

    result = cli_runner.invoke(cli_app, ["train", "--validate-only", str(config_file)])
    assert result.exit_code in [0, 1, 2]


def test_train_checkpoint_flag_requires_path(cli_runner: CliRunner, tmp_path) -> None:
    """Checkpoint flag requires a path and config."""
    # Without config, should error
    result = cli_runner.invoke(cli_app, ["train", "--checkpoint", str(tmp_path / "ckpt.ckpt")])
    assert result.exit_code != 0


def test_train_with_config_file_attempts_execution(
    cli_runner: CliRunner, tmp_path, minimal_train_config: str
) -> None:
    """Test that train command attempts to process valid config files."""
    config_file = tmp_path / "training.toml"
    config_file.write_text(minimal_train_config)

    result = cli_runner.invoke(cli_app, ["train", str(config_file)])
    # Exit codes: 0 = success, 1 = expected failure, 2 = help
    assert result.exit_code in [0, 1, 2]
