"""Error-path tests for the CLI train command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import typer
from typer.testing import CliRunner

from dlkit.common.errors import ConfigurationError, WorkflowError
from dlkit.interfaces.cli.app import app as cli_app


class TestTrainCommandErrorHandling:
    """Test error handling scenarios for train command."""

    def test_train_missing_config_file(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        nonexistent_config = tmp_path / "nonexistent.toml"

        result = cli_runner.invoke(cli_app, ["train", str(nonexistent_config)])

        assert result.exit_code == 1

    def test_train_validate_only_flag_requires_config(
        self,
        cli_runner: CliRunner,
    ) -> None:
        result = cli_runner.invoke(cli_app, ["train", "--validate-only"])

        assert result.exit_code != 0

    def test_train_checkpoint_flag_requires_config(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        result = cli_runner.invoke(
            cli_app,
            ["train", "--checkpoint", str(tmp_path / "ckpt.ckpt")],
        )

        assert result.exit_code != 0

    def test_train_config_loading_error(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")

        with (
            patch(
                "dlkit.interfaces.cli.commands.train.load_config",
                side_effect=ConfigurationError("Invalid config", {}),
            ),
            patch(
                "dlkit.interfaces.cli.middleware.error_handler.handle_api_error"
            ) as mock_handle_error,
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
        config_path = tmp_path / "config.toml"
        config_path.write_text("[SESSION]\nname = 'test'")
        mock_settings = mock_settings_factory("vanilla")

        with (
            patch("dlkit.interfaces.cli.commands.train.load_config", return_value=mock_settings),
            patch(
                "dlkit.interfaces.cli.commands.train.validate_config",
                side_effect=ConfigurationError("Invalid config", {}),
            ),
            patch(
                "dlkit.interfaces.cli.middleware.error_handler.handle_api_error"
            ) as mock_handle_error,
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
            patch(
                "dlkit.interfaces.cli.middleware.error_handler.handle_api_error"
            ) as mock_handle_error,
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

    def test_train_typer_exit_preservation(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        mock_settings_factory,
    ) -> None:
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
