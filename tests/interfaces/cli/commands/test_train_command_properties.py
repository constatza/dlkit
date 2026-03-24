"""Boundary-focused train command tests for the CLI layer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from dlkit.interfaces.cli.app import app as cli_app


def _mock_train_settings() -> Mock:
    settings = Mock()
    settings.MLFLOW = Mock(is_active=False)
    settings.OPTUNA = Mock(is_active=False)
    return settings


@pytest.mark.parametrize("epochs", [1, 2, 1000])
def test_train_numeric_parameter_ranges(
    epochs: int,
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[SESSION]\nname = 'test'")

    with (
        patch(
            "dlkit.interfaces.cli.commands.train.load_config",
            return_value=_mock_train_settings(),
        ),
        patch("dlkit.interfaces.cli.commands.train.validate_config"),
        patch("dlkit.interfaces.cli.commands.train.api_train", return_value=Mock()),
        patch("dlkit.interfaces.cli.commands.train.present_training_result"),
    ):
        result = cli_runner.invoke(cli_app, ["train", "--epochs", str(epochs), str(config_path)])

    assert result.exit_code == 0


@pytest.mark.parametrize(
    ("experiment_name", "run_name"),
    [
        ("exp-1024", "run-1024"),
        ("exp-5000", "run-5000"),
        ("exp-65535", "run-65535"),
    ],
)
def test_train_mlflow_parameters(
    experiment_name: str,
    run_name: str,
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[SESSION]\nname = 'test'")

    with (
        patch(
            "dlkit.interfaces.cli.commands.train.load_config",
            return_value=_mock_train_settings(),
        ),
        patch("dlkit.interfaces.cli.commands.train.validate_config"),
        patch("dlkit.interfaces.cli.commands.train.api_train", return_value=Mock()),
        patch("dlkit.interfaces.cli.commands.train.present_training_result"),
    ):
        result = cli_runner.invoke(
            cli_app,
            [
                "train",
                "--mlflow",
                "--experiment-name",
                experiment_name,
                "--run-name",
                run_name,
                str(config_path),
            ],
        )

    assert result.exit_code == 0
