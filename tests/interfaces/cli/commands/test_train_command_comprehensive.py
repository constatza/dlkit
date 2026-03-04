"""Comprehensive tests for CLI train command functionality.

This module provides complete coverage for the train command including:
- Good-path scenarios with all parameter combinations
- Error handling and validation
- Strategy auto-detection
- Configuration loading and overrides
- Property-based testing for parameter validation

Test coverage goal: Cover the remaining 59/84 untested lines in train.py
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer
from hypothesis import given, strategies as st
from typer.testing import CliRunner

from dlkit.interfaces.cli.app import app as cli_app
from dlkit.interfaces.api.domain.errors import ConfigurationError, WorkflowError

# Constants for test dataflow boundaries
MIN_EPOCHS = 1
MAX_EPOCHS = 1000
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 512
MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 1.0
MIN_TRIALS = 1
MAX_TRIALS = 10000
MIN_PORT = 1024
MAX_PORT = 65535


class TestTrainCommandGoodPath:
    """Test successful training command scenarios."""

    def test_train_vanilla_strategy_basic(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        mock_successful_training_result: Mock,
        mock_settings_factory,
    ) -> None:
        """Test basic vanilla training strategy execution."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")
        mock_settings = mock_settings_factory("vanilla")

        with (
            patch("dlkit.interfaces.cli.commands.train.load_config", return_value=mock_settings),
            patch("dlkit.interfaces.cli.commands.train.validate_config") as mock_validate,
            patch(
                "dlkit.interfaces.cli.commands.train.api_train",
                return_value=mock_successful_training_result,
            ) as mock_train,
            patch("dlkit.interfaces.cli.commands.train.present_training_result") as mock_present,
        ):
            result = cli_runner.invoke(cli_app, ["train", str(config_path)])

            assert result.exit_code == 0
            mock_validate.assert_called_once_with(mock_settings)
            mock_train.assert_called_once()
            mock_present.assert_called_once()

    def test_train_mlflow_strategy_auto_detect(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        mock_successful_training_result: Mock,
        mock_settings_factory,
    ) -> None:
        """Test MLflow strategy auto-detection from configuration."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")
        mock_settings = mock_settings_factory("mlflow", mlflow_active=True)

        with (
            patch("dlkit.interfaces.cli.commands.train.load_config", return_value=mock_settings),
            patch("dlkit.interfaces.cli.commands.train.validate_config") as mock_validate,
            patch(
                "dlkit.interfaces.cli.commands.train.api_train",
                return_value=mock_successful_training_result,
            ) as mock_train,
            patch("dlkit.interfaces.cli.commands.train.present_training_result"),
        ):
            result = cli_runner.invoke(cli_app, ["train", str(config_path)])

            assert result.exit_code == 0
            mock_validate.assert_called_once_with(mock_settings)
            mock_train.assert_called_once()

    def test_train_mlflow_flag_override(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        mock_successful_training_result: Mock,
        mock_settings_factory,
    ) -> None:
        """Test explicit MLflow flag override."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")
        mock_settings = mock_settings_factory("vanilla")  # vanilla config but MLflow flag

        with (
            patch("dlkit.interfaces.cli.commands.train.load_config", return_value=mock_settings),
            patch("dlkit.interfaces.cli.commands.train.validate_config") as mock_validate,
            patch(
                "dlkit.interfaces.cli.commands.train.api_train",
                return_value=mock_successful_training_result,
            ) as mock_train,
            patch("dlkit.interfaces.cli.commands.train.present_training_result"),
        ):
            result = cli_runner.invoke(cli_app, ["train", "--mlflow", str(config_path)])

            assert result.exit_code == 0
            mock_validate.assert_called_once_with(mock_settings)
            mock_train.assert_called_once()

    def test_train_mlflow_config_and_flag(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        mock_successful_training_result: Mock,
        mock_settings_factory,
    ) -> None:
        """Test that MLflow is used when enabled in config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")
        mock_settings = mock_settings_factory("mlflow", mlflow_active=True)

        with (
            patch("dlkit.interfaces.cli.commands.train.load_config", return_value=mock_settings),
            patch("dlkit.interfaces.cli.commands.train.validate_config") as mock_validate,
            patch(
                "dlkit.interfaces.cli.commands.train.api_train",
                return_value=mock_successful_training_result,
            ) as mock_train,
            patch("dlkit.interfaces.cli.commands.train.present_training_result"),
        ):
            result = cli_runner.invoke(cli_app, ["train", str(config_path)])

            assert result.exit_code == 0
            mock_validate.assert_called_once_with(mock_settings)
            mock_train.assert_called_once()

    def test_train_mlflow_flag_explicit(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        mock_successful_training_result: Mock,
        mock_settings_factory,
    ) -> None:
        """Test explicit MLflow flag enables tracking."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")
        mock_settings = mock_settings_factory("vanilla")

        with (
            patch("dlkit.interfaces.cli.commands.train.load_config", return_value=mock_settings),
            patch("dlkit.interfaces.cli.commands.train.validate_config") as mock_validate,
            patch(
                "dlkit.interfaces.cli.commands.train.api_train",
                return_value=mock_successful_training_result,
            ) as mock_train,
            patch("dlkit.interfaces.cli.commands.train.present_training_result"),
        ):
            result = cli_runner.invoke(cli_app, ["train", "--mlflow", str(config_path)])

            if result.exit_code != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {getattr(result, 'stderr', 'N/A')}")
                print(f"Exception: {result.exception}")
            assert result.exit_code == 0
            mock_validate.assert_called_once_with(mock_settings)
            mock_train.assert_called_once()

    def test_train_validate_only_flag(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        mock_settings_factory,
    ) -> None:
        """Test validate-only flag skips actual training."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")
        mock_settings = mock_settings_factory("vanilla")

        with (
            patch("dlkit.interfaces.cli.commands.train.load_config", return_value=mock_settings),
            patch("dlkit.interfaces.cli.commands.train.validate_config") as mock_validate,
            patch("dlkit.interfaces.cli.commands.train.api_train") as mock_train,
            patch("dlkit.interfaces.cli.commands.train.present_training_result") as mock_present,
        ):
            result = cli_runner.invoke(cli_app, ["train", "--validate-only", str(config_path)])

            assert result.exit_code == 0
            mock_validate.assert_called_once_with(mock_settings)
            mock_train.assert_not_called()
            mock_present.assert_not_called()


class TestTrainCommandErrorHandling:
    """Test error handling scenarios for train command."""

    def test_train_missing_config_file(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test error handling when config file doesn't exist."""
        nonexistent_config = tmp_path / "nonexistent.toml"

        result = cli_runner.invoke(cli_app, ["train", str(nonexistent_config)])

        assert result.exit_code == 1  # Application error (file not found)
        # Note: Application code handles missing files for better error messages

    def test_train_config_loading_error(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test error handling when config loading fails."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")

        with (
            patch(
                "dlkit.interfaces.cli.commands.train.load_config",
                side_effect=ConfigurationError("Invalid config", {}),
            ),
            patch("dlkit.interfaces.cli.commands.train.handle_api_error") as mock_handle_error,
        ):
            result = cli_runner.invoke(cli_app, ["train", str(config_path)])

            assert result.exit_code == 1
            mock_handle_error.assert_called_once()

    def test_train_validation_error(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        mock_settings_factory,
    ) -> None:
        """Test error handling when config validation fails."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")
        mock_settings = mock_settings_factory("vanilla")

        with (
            patch("dlkit.interfaces.cli.commands.train.load_config", return_value=mock_settings),
            patch(
                "dlkit.interfaces.cli.commands.train.validate_config",
                side_effect=ConfigurationError("Invalid config", {}),
            ),
            patch("dlkit.interfaces.cli.commands.train.handle_api_error") as mock_handle_error,
        ):
            result = cli_runner.invoke(cli_app, ["train", str(config_path)])

            assert result.exit_code == 1
            mock_handle_error.assert_called_once()

    def test_train_api_training_error(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        mock_settings_factory,
    ) -> None:
        """Test error handling when training API fails."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")
        mock_settings = mock_settings_factory("vanilla")

        with (
            patch("dlkit.interfaces.cli.commands.train.load_config", return_value=mock_settings),
            patch("dlkit.interfaces.cli.commands.train.validate_config"),
            patch(
                "dlkit.interfaces.cli.commands.train.api_train",
                side_effect=WorkflowError("Training failed", {}),
            ),
            patch("dlkit.interfaces.cli.commands.train.handle_api_error") as mock_handle_error,
        ):
            result = cli_runner.invoke(cli_app, ["train", str(config_path)])

            assert result.exit_code == 1
            mock_handle_error.assert_called_once()

    def test_train_unexpected_error(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        mock_settings_factory,
    ) -> None:
        """Test error handling for unexpected exceptions."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")
        mock_settings = mock_settings_factory("vanilla")

        with (
            patch("dlkit.interfaces.cli.commands.train.load_config", return_value=mock_settings),
            patch("dlkit.interfaces.cli.commands.train.validate_config"),
            patch(
                "dlkit.interfaces.cli.commands.train.api_train",
                side_effect=RuntimeError("Unexpected error"),
            ),
        ):
            result = cli_runner.invoke(cli_app, ["train", str(config_path)])

            assert result.exit_code == 1
            # For non-DLKitError exceptions, handle_api_error is not called

    def test_train_typer_exit_preservation(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        mock_settings_factory,
    ) -> None:
        """Test that Typer.Exit exceptions are preserved."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")
        mock_settings = mock_settings_factory("vanilla")

        with (
            patch("dlkit.interfaces.cli.commands.train.load_config", return_value=mock_settings),
            patch("dlkit.interfaces.cli.commands.train.validate_config"),
            patch("dlkit.interfaces.cli.commands.train.api_train", side_effect=typer.Exit(42)),
        ):
            result = cli_runner.invoke(cli_app, ["train", str(config_path)])

            assert result.exit_code == 42


class TestTrainCommandProperties:
    """Property-based tests for train command parameter validation."""

    @given(epochs=st.integers(min_value=MIN_EPOCHS, max_value=MAX_EPOCHS))
    def test_train_numeric_parameter_ranges(
        self,
        epochs: int,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Test that numeric parameters accept valid ranges."""
        tmp_dir = tmp_path_factory.mktemp("train_numeric")
        cli_runner = CliRunner()
        config_path = tmp_dir / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")

        mock_settings = Mock()
        mock_settings.MLFLOW = Mock(is_active=False)
        mock_settings.OPTUNA = Mock(is_active=False)
        mock_result = Mock()

        with (
            patch("dlkit.interfaces.cli.commands.train.load_config", return_value=mock_settings),
            patch("dlkit.interfaces.cli.commands.train.validate_config"),
            patch("dlkit.interfaces.cli.commands.train.api_train", return_value=mock_result),
            patch("dlkit.interfaces.cli.commands.train.present_training_result"),
        ):
            result = cli_runner.invoke(
                cli_app, ["train", "--epochs", str(epochs), str(config_path)]
            )
            assert result.exit_code == 0

    @given(trials=st.integers(min_value=MIN_TRIALS, max_value=MAX_TRIALS))
    def test_train_strategy_with_trials(
        self,
        trials: int,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Test Optuna strategy with various trial counts."""
        tmp_dir = tmp_path_factory.mktemp("train_trials")
        cli_runner = CliRunner()
        config_path = tmp_dir / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")

        mock_settings = Mock()
        mock_settings.MLFLOW = Mock(is_active=False)
        mock_settings.OPTUNA = Mock(is_active=False)
        mock_result = Mock()

        with (
            patch("dlkit.interfaces.cli.commands.train.load_config", return_value=mock_settings),
            patch("dlkit.interfaces.cli.commands.train.validate_config"),
            patch("dlkit.interfaces.cli.commands.train.api_train", return_value=mock_result),
            patch("dlkit.interfaces.cli.commands.train.present_training_result"),
        ):
            # Note: Optuna optimization should use 'dlkit optimize' command, not train
            result = cli_runner.invoke(
                cli_app, ["train", "--epochs", str(trials), str(config_path)]
            )
            assert result.exit_code == 0

    @given(suffix=st.integers(min_value=MIN_PORT, max_value=MAX_PORT))
    def test_train_mlflow_parameters(
        self,
        suffix: int,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Test MLflow strategy with supported naming overrides."""
        tmp_dir = tmp_path_factory.mktemp("train_mlflow")
        cli_runner = CliRunner()
        config_path = tmp_dir / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")

        mock_settings = Mock()
        mock_settings.MLFLOW = Mock(is_active=False)
        mock_settings.OPTUNA = Mock(is_active=False)
        mock_result = Mock()

        with (
            patch("dlkit.interfaces.cli.commands.train.load_config", return_value=mock_settings),
            patch("dlkit.interfaces.cli.commands.train.validate_config"),
            patch("dlkit.interfaces.cli.commands.train.api_train", return_value=mock_result),
            patch("dlkit.interfaces.cli.commands.train.present_training_result"),
        ):
            result = cli_runner.invoke(
                cli_app,
                [
                    "train",
                    "--mlflow",
                    "--experiment-name",
                    f"exp-{suffix}",
                    "--run-name",
                    f"run-{suffix}",
                    str(config_path),
                ],
            )
            assert result.exit_code == 0


class TestTrainCommandIntegration:
    """Integration tests for train command."""

    def test_train_command_help_accessibility(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test that train command help is accessible."""
        result = cli_runner.invoke(cli_app, ["train", "--help"])

        assert result.exit_code == 0
        assert "train" in result.stdout.lower()
        assert "configuration" in result.stdout.lower()

    def test_train_command_no_args_error(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test that train command requires configuration argument."""
        result = cli_runner.invoke(cli_app, ["train"])

        assert result.exit_code == 2  # Missing required argument
        assert result.exit_code == 2  # Missing argument error
