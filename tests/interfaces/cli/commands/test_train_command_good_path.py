"""Good-path tests for the CLI train command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from dlkit.interfaces.cli.app import app as cli_app


class TestTrainCommandGoodPath:
    """Test successful training command scenarios."""

    def test_train_vanilla_strategy_basic(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        mock_successful_training_result: Mock,
        mock_settings_factory,
    ) -> None:
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

        assert result.exit_code == 0, result.output
        mock_validate.assert_called_once_with(mock_settings)
        mock_train.assert_called_once()

    def test_train_validate_only_flag(
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
            patch("dlkit.interfaces.cli.commands.train.validate_config") as mock_validate,
            patch("dlkit.interfaces.cli.commands.train.api_train") as mock_train,
            patch("dlkit.interfaces.cli.commands.train.present_training_result") as mock_present,
        ):
            result = cli_runner.invoke(cli_app, ["train", "--validate-only", str(config_path)])

        assert result.exit_code == 0
        mock_validate.assert_called_once_with(mock_settings)
        mock_train.assert_not_called()
        mock_present.assert_not_called()
