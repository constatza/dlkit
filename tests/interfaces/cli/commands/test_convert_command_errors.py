"""Error-path tests for the CLI convert command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from dlkit.common.errors import WorkflowError
from dlkit.interfaces.cli.app import app as cli_app


class TestConvertCommandErrors:
    """Test error handling scenarios for convert command."""

    def test_missing_both_shape_and_config(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        result = cli_runner.invoke(
            cli_app, ["convert", "entry", str(checkpoint_path), str(output_path)]
        )

        assert result.exit_code == 1

    def test_conversion_command_failure(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        with patch(
            "dlkit.interfaces.cli.commands.convert.convert_checkpoint_to_onnx"
        ) as mock_convert:
            mock_convert.side_effect = WorkflowError(
                "Checkpoint file not found", {"workflow": "convert"}
            )
            result = cli_runner.invoke(
                cli_app,
                [
                    "convert",
                    "entry",
                    str(checkpoint_path),
                    str(output_path),
                    "--shape",
                    "3,224,224",
                ],
            )

        assert result.exit_code == 1

    def test_config_loading_failure(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"
        config_path = tmp_path / "invalid_config.toml"
        config_path.write_text("invalid toml content [")

        with patch("dlkit.interfaces.cli.commands.convert.load_config") as mock_load:
            mock_load.side_effect = Exception("Invalid configuration format")
            result = cli_runner.invoke(
                cli_app,
                [
                    "convert",
                    "entry",
                    str(checkpoint_path),
                    str(output_path),
                    "--config",
                    str(config_path),
                ],
            )

        assert result.exit_code == 1

    def test_generic_exception_handling(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        with patch(
            "dlkit.interfaces.cli.commands.convert.convert_checkpoint_to_onnx"
        ) as mock_convert:
            mock_convert.side_effect = RuntimeError("Unexpected system error")
            result = cli_runner.invoke(
                cli_app,
                [
                    "convert",
                    "entry",
                    str(checkpoint_path),
                    str(output_path),
                    "--shape",
                    "3,224,224",
                ],
            )

        assert result.exit_code == 1
