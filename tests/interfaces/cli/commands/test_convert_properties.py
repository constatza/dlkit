"""Boundary-focused tests for the CLI convert command.

The API layer already owns the exhaustive Hypothesis coverage for convert
validation and shape parsing. The CLI layer only needs to prove argument
wiring, error surfacing, and a few representative boundaries.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from dlkit.interfaces.cli.app import app as cli_app

MIN_OPSET = 9
MAX_OPSET = 20


def _write_convert_paths(tmp_path: Path, checkpoint_name: str = "model.ckpt") -> tuple[Path, Path]:
    checkpoint_path = tmp_path / checkpoint_name
    checkpoint_path.write_text("dummy checkpoint")
    output_path = tmp_path / "model.onnx"
    return checkpoint_path, output_path


class TestConvertCommandProperties:
    """Boundary tests for successful CLI argument handling."""

    @pytest.mark.parametrize(
        ("shape_str", "batch_size", "opset"),
        [
            ("1", 1, MIN_OPSET),
            ("3,224,224", 8, 13),
            ("32,64,128,256,512", 512, MAX_OPSET),
        ],
    )
    def test_valid_shape_parameters_property(
        self,
        shape_str: str,
        batch_size: int,
        opset: int,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        checkpoint_path, output_path = _write_convert_paths(tmp_path)

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
                    shape_str,
                    "--batch-size",
                    str(batch_size),
                    "--opset",
                    str(opset),
                ],
            )

        assert result.exit_code == 0
        input_data = mock_cmd.execute.call_args[0][0]
        assert input_data.shape == shape_str
        assert input_data.batch_size == batch_size
        assert input_data.opset == opset

    @pytest.mark.parametrize("filename_base", ["model", "model_1", "Model123", "a" * 40])
    def test_valid_filename_handling_property(
        self,
        filename_base: str,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        checkpoint_path = tmp_path / f"{filename_base}.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / f"{filename_base}_output.onnx"

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
        input_data = mock_cmd.execute.call_args[0][0]
        assert input_data.checkpoint_path == checkpoint_path
        assert input_data.output_path == output_path

    @pytest.mark.parametrize(
        "multi_shape",
        [
            "4;8",
            "3,224,224;1,128",
            "8,16;4,8,12;2,3,5,7",
        ],
    )
    def test_multiple_input_shapes_property(
        self,
        multi_shape: str,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        checkpoint_path, output_path = _write_convert_paths(tmp_path)

        from dlkit.interfaces.api.commands.convert_command import ConvertResult

        multi_result = ConvertResult(
            output_path=sample_convert_result.output_path,
            opset=sample_convert_result.opset,
            inputs=[(1, 4), (1, 8)],
        )

        with patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls:
            mock_cmd = Mock()
            mock_cmd.execute.return_value = multi_result
            mock_cmd_cls.return_value = mock_cmd

            result = cli_runner.invoke(
                cli_app,
                [
                    "convert",
                    "entry",
                    str(checkpoint_path),
                    str(output_path),
                    "--shape",
                    multi_shape,
                ],
            )

        assert result.exit_code == 0
        input_data = mock_cmd.execute.call_args[0][0]
        assert input_data.shape == multi_shape

    @pytest.mark.parametrize("opset", [MIN_OPSET, 13, MAX_OPSET])
    def test_opset_version_boundary_property(
        self,
        opset: int,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        checkpoint_path, output_path = _write_convert_paths(tmp_path)

        from dlkit.interfaces.api.commands.convert_command import ConvertResult

        opset_result = ConvertResult(
            output_path=sample_convert_result.output_path,
            opset=opset,
            inputs=sample_convert_result.inputs,
        )

        with patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls:
            mock_cmd = Mock()
            mock_cmd.execute.return_value = opset_result
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
                    "--opset",
                    str(opset),
                ],
            )

        assert result.exit_code == 0
        assert f"Opset: {opset}" in result.stdout
        assert mock_cmd.execute.call_args[0][0].opset == opset

    @pytest.mark.parametrize("batch_size", [1, 8, 512])
    def test_batch_size_boundary_property(
        self,
        batch_size: int,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        checkpoint_path, output_path = _write_convert_paths(tmp_path)

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
                    "784",
                    "--batch-size",
                    str(batch_size),
                ],
            )

        assert result.exit_code == 0
        assert mock_cmd.execute.call_args[0][0].batch_size == batch_size


class TestConvertCommandInvalidProperties:
    """Tests for invalid parameter handling."""

    @pytest.mark.parametrize(
        "invalid_shape",
        ["", "0,224,224", "-1,224,224", "abc,224,224", "3,224,", ",224,224", "3,,224"],
    )
    def test_invalid_shape_handling_property(
        self,
        invalid_shape: str,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        checkpoint_path, output_path = _write_convert_paths(tmp_path)

        from dlkit.interfaces.api.domain.errors import WorkflowError

        with patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls:
            mock_cmd = Mock()
            mock_cmd.execute.side_effect = WorkflowError(
                f"Invalid shape spec: '{invalid_shape}'", {"shape": invalid_shape}
            )
            mock_cmd_cls.return_value = mock_cmd

            result = cli_runner.invoke(
                cli_app,
                [
                    "convert",
                    "entry",
                    str(checkpoint_path),
                    str(output_path),
                    "--shape",
                    invalid_shape,
                ],
            )

        assert result.exit_code == 1
        assert "Export failed:" in result.stdout

    @pytest.mark.parametrize("opset", [0, 1, 8])
    def test_invalid_opset_property(
        self,
        opset: int,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        checkpoint_path, output_path = _write_convert_paths(tmp_path)

        from dlkit.interfaces.api.domain.errors import WorkflowError

        with patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls:
            mock_cmd = Mock()
            mock_cmd.execute.side_effect = WorkflowError(
                "Unsupported opset version (min 9)", {"command": "convert", "opset": opset}
            )
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
                    "--opset",
                    str(opset),
                ],
            )

        assert result.exit_code == 1
        assert "Export failed:" in result.stdout


class TestConvertCommandRobustness:
    """Robustness tests for representative edge-case combinations."""

    @pytest.mark.parametrize("path_length", [1, 10, 50, 100])
    def test_path_length_robustness(
        self,
        path_length: int,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        filename_base = "a" * min(path_length, 50)
        checkpoint_path = tmp_path / f"{filename_base}.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / f"{filename_base}_out.onnx"

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

    @pytest.mark.parametrize(
        "args",
        [
            ["--shape", "3,224,224"],
            ["--shape", "16,32", "--batch-size", "4"],
            ["--shape", "8,16;4,8", "--batch-size", "2", "--opset", "13"],
        ],
    )
    def test_parameter_combination_robustness(
        self,
        args: list[str],
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        checkpoint_path, output_path = _write_convert_paths(tmp_path)

        with patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls:
            mock_cmd = Mock()
            mock_cmd.execute.return_value = sample_convert_result
            mock_cmd_cls.return_value = mock_cmd

            result = cli_runner.invoke(
                cli_app,
                ["convert", "entry", str(checkpoint_path), str(output_path), *args],
            )

        assert result.exit_code == 0
