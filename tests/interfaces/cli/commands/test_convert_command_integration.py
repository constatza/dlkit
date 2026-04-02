"""Integration-facing checks for convert CLI path and argument handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from dlkit.interfaces.cli.app import app as cli_app


def test_path_handling_with_spaces(
    cli_runner: CliRunner,
    tmp_path: Path,
    sample_convert_result: Mock,
) -> None:
    checkpoint_path = tmp_path / "model checkpoint.ckpt"
    checkpoint_path.write_text("dummy checkpoint")
    output_path = tmp_path / "output model.onnx"

    with patch("dlkit.interfaces.cli.commands.convert.convert_checkpoint_to_onnx") as mock_convert:
        mock_convert.return_value = sample_convert_result
        result = cli_runner.invoke(
            cli_app,
            ["convert", "entry", str(checkpoint_path), str(output_path), "--shape", "3,224,224"],
        )

    assert result.exit_code == 0
    assert mock_convert.call_args.kwargs["checkpoint_path"] == checkpoint_path
    assert mock_convert.call_args.kwargs["output_path"] == output_path


def test_multi_input_shape_is_forwarded(
    cli_runner: CliRunner,
    tmp_path: Path,
    sample_convert_result: Mock,
) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_text("dummy checkpoint")
    output_path = tmp_path / "model.onnx"

    with patch("dlkit.interfaces.cli.commands.convert.convert_checkpoint_to_onnx") as mock_convert:
        mock_convert.return_value = sample_convert_result
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
    assert mock_convert.call_args.kwargs["shape"] == "3,224,224;512"
