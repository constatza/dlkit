"""Good-path tests for the CLI convert command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from dlkit.interfaces.cli.app import app as cli_app


class TestConvertCommandGoodPath:
    """Test successful convert command scenarios."""

    def test_convert_with_shape_basic(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        with patch(
            "dlkit.interfaces.cli.commands.convert.convert_checkpoint_to_onnx"
        ) as mock_convert:
            mock_convert.return_value = sample_convert_result
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

        assert result.exit_code == 0
        assert mock_convert.call_args.kwargs["checkpoint_path"] == checkpoint_path
        assert mock_convert.call_args.kwargs["output_path"] == output_path
        assert mock_convert.call_args.kwargs["shape"] == "3,224,224"
        assert mock_convert.call_args.kwargs["batch_size"] == 1
        assert mock_convert.call_args.kwargs["opset"] == 17

    def test_convert_with_config_file(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
        sample_config_file: Path,
        sample_settings: Mock,
    ) -> None:
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        with (
            patch(
                "dlkit.interfaces.cli.commands.convert.convert_checkpoint_to_onnx"
            ) as mock_convert,
            patch(
                "dlkit.interfaces.cli.commands.convert.load_config",
                return_value=sample_settings,
            ) as mock_load,
        ):
            mock_convert.return_value = sample_convert_result
            result = cli_runner.invoke(
                cli_app,
                [
                    "convert",
                    "entry",
                    str(checkpoint_path),
                    str(output_path),
                    "--config",
                    str(sample_config_file),
                ],
            )

        assert result.exit_code == 0
        mock_load.assert_called_once_with(sample_config_file)
        assert mock_convert.call_args.kwargs["shape"] is None
        assert mock_convert.call_args.kwargs["settings"] == sample_settings

    def test_convert_with_config_and_batch_validation(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
        sample_config_file: Path,
        sample_settings: Mock,
    ) -> None:
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        with (
            patch(
                "dlkit.interfaces.cli.commands.convert.convert_checkpoint_to_onnx"
            ) as mock_convert,
            patch(
                "dlkit.interfaces.cli.commands.convert.load_config", return_value=sample_settings
            ),
        ):
            mock_convert.return_value = sample_convert_result
            result = cli_runner.invoke(
                cli_app,
                [
                    "convert",
                    "entry",
                    str(checkpoint_path),
                    str(output_path),
                    "--config",
                    str(sample_config_file),
                    "--batch-size",
                    "8",
                ],
            )

        assert result.exit_code == 0
        assert mock_convert.call_args.kwargs["batch_size"] == 8

    def test_convert_short_options(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
        sample_config_file: Path,
        sample_settings: Mock,
    ) -> None:
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        with (
            patch(
                "dlkit.interfaces.cli.commands.convert.convert_checkpoint_to_onnx"
            ) as mock_convert,
            patch(
                "dlkit.interfaces.cli.commands.convert.load_config", return_value=sample_settings
            ),
        ):
            mock_convert.return_value = sample_convert_result
            result = cli_runner.invoke(
                cli_app,
                [
                    "convert",
                    "entry",
                    str(checkpoint_path),
                    str(output_path),
                    "-c",
                    str(sample_config_file),
                    "-b",
                    "16",
                ],
            )

        assert result.exit_code == 0
        assert mock_convert.call_args.kwargs["batch_size"] == 16
