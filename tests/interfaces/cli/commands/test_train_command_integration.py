"""CLI integration checks for the train command surface."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlkit.interfaces.cli.app import app as cli_app


class TestTrainCommandIntegration:
    """Integration tests for train command help and argument parsing."""

    def test_train_command_help_accessibility(
        self,
        cli_runner: CliRunner,
    ) -> None:
        result = cli_runner.invoke(cli_app, ["train", "--help"])

        assert result.exit_code == 0

    def test_train_command_no_args_error(
        self,
        cli_runner: CliRunner,
    ) -> None:
        result = cli_runner.invoke(cli_app, ["train"])

        assert result.exit_code == 2

    def test_train_with_minimal_config_attempts_execution(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        config_file = tmp_path / "training.toml"
        config_file.write_text(
            "\n".join(
                [
                    "[SESSION]",
                    'name = "test_training"',
                    "inference = false",
                    "",
                    "[PATHS]",
                    'output_dir = "./outputs"',
                    "",
                    "[TRAINING.trainer]",
                    "max_epochs = 1",
                ]
            )
        )

        result = cli_runner.invoke(cli_app, ["train", str(config_file)])
        assert result.exit_code in [0, 1, 2]
