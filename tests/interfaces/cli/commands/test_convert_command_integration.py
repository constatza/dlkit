"""Integration-facing checks for convert CLI presentation and path handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from dlkit.interfaces.api.commands.convert_command import ConvertResult
from dlkit.interfaces.cli.app import app as cli_app


def test_path_handling_with_spaces(
    cli_runner: CliRunner,
    tmp_path: Path,
    sample_convert_result: Mock,
) -> None:
    checkpoint_path = tmp_path / "model checkpoint.ckpt"
    checkpoint_path.write_text("dummy checkpoint")
    output_path = tmp_path / "output model.onnx"

    with patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls:
        mock_cmd = Mock()
        mock_cmd.execute.return_value = sample_convert_result
        mock_cmd_cls.return_value = mock_cmd

        result = cli_runner.invoke(
            cli_app,
            ["convert", "entry", str(checkpoint_path), str(output_path), "--shape", "3,224,224"],
        )

    assert result.exit_code == 0
    input_data = mock_cmd.execute.call_args[0][0]
    assert input_data.checkpoint_path == checkpoint_path
    assert input_data.output_path == output_path


def test_console_output_formatting(
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_text("dummy checkpoint")
    output_path = tmp_path / "model.onnx"

    detailed_result = ConvertResult(
        output_path=Path("/path/to/exported/model.onnx"),
        opset=17,
        inputs=[(1, 3, 224, 224), (1, 512)],
    )

    with patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls:
        mock_cmd = Mock()
        mock_cmd.execute.return_value = detailed_result
        mock_cmd_cls.return_value = mock_cmd

        result = cli_runner.invoke(
            cli_app,
            [
                "convert",
                "entry",
                str(checkpoint_path),
                str(output_path),
                "--shape",
                "3,224,224;512",
            ],
        )

    assert result.exit_code == 0
    assert "✅ Export successful" in result.stdout
    assert "Output: /path/to/exported/model.onnx" in result.stdout
    assert "Opset: 17" in result.stdout
    assert "Inputs: (1, 3, 224, 224), (1, 512)" in result.stdout
    assert "ONNX Export" in result.stdout
