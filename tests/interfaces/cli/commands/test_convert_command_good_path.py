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

        with patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls:
            mock_cmd = Mock()
            mock_cmd.execute.return_value = sample_convert_result
            mock_cmd_cls.return_value = mock_cmd

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
        assert "✅ Export successful" in result.stdout
        assert "test_model.onnx" in result.stdout
        assert "Opset: 17" in result.stdout
        assert "Inputs: (1, 3, 224, 224)" in result.stdout

        input_data = mock_cmd.execute.call_args[0][0]
        assert input_data.checkpoint_path == checkpoint_path
        assert input_data.output_path == output_path
        assert input_data.shape == "3,224,224"
        assert input_data.batch_size == 1
        assert input_data.opset == 17

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
            patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls,
            patch(
                "dlkit.interfaces.cli.commands.convert.load_config",
                return_value=sample_settings,
            ) as mock_load,
        ):
            mock_cmd = Mock()
            mock_cmd.execute.return_value = sample_convert_result
            mock_cmd_cls.return_value = mock_cmd

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
        input_data, settings = mock_cmd.execute.call_args[0]
        assert input_data.shape is None
        assert settings == sample_settings

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
            patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls,
            patch(
                "dlkit.interfaces.cli.commands.convert.load_config", return_value=sample_settings
            ),
        ):
            mock_cmd = Mock()
            mock_cmd.execute.return_value = sample_convert_result
            mock_cmd_cls.return_value = mock_cmd

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
        assert mock_cmd.execute.call_args[0][0].batch_size == 8

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
            patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls,
            patch(
                "dlkit.interfaces.cli.commands.convert.load_config", return_value=sample_settings
            ),
        ):
            mock_cmd = Mock()
            mock_cmd.execute.return_value = sample_convert_result
            mock_cmd_cls.return_value = mock_cmd

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
        assert mock_cmd.execute.call_args[0][0].batch_size == 16
