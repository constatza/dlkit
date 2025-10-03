"""Comprehensive tests for CLI convert command functionality.

This module provides complete coverage for the convert command including:
- Good-path scenarios with all parameter combinations
- Error handling and validation
- CLI argument parsing and validation
- Integration with ConvertCommand execution
- Rich console output formatting

Test coverage goal: Achieve 100% coverage of convert.py (29 lines total)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from dlkit.interfaces.cli.app import app as cli_app
from dlkit.interfaces.api.domain.errors import WorkflowError

# Constants for test dataflow boundaries
MIN_OPSET = 9
MAX_OPSET = 20
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 1024


class TestConvertCommandGoodPath:
    """Test successful convert command scenarios."""

    def test_convert_with_shape_basic(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        """Test basic conversion with shape parameter.

        Args:
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
            sample_convert_result: Mock ConvertResult fixture.
        """
        # Create required files
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

            # Verify command was called with correct parameters
            mock_cmd.execute.assert_called_once()
            call_args = mock_cmd.execute.call_args
            input_data = call_args[0][0]

            assert input_data.checkpoint_path == checkpoint_path
            assert input_data.output_path == output_path
            assert input_data.shape == "3,224,224"
            assert input_data.batch_size == 1
            assert input_data.opset == 17

    def test_convert_with_shape_and_batch_size(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        """Test conversion with shape and custom batch size.

        Args:
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
            sample_convert_result: Mock ConvertResult fixture.
        """
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
                    "3,32,32",
                    "--batch-size",
                    "4",
                ],
            )

            assert result.exit_code == 0

            # Verify command was called with correct batch size
            call_args = mock_cmd.execute.call_args
            input_data = call_args[0][0]
            assert input_data.batch_size == 4
            assert input_data.shape == "3,32,32"

    def test_convert_with_config_file(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
        sample_config_file: Path,
        sample_settings: Mock,
    ) -> None:
        """Test conversion using config file for shape inference.

        Args:
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
            sample_convert_result: Mock ConvertResult fixture.
            sample_config_file: Sample configuration file fixture.
            sample_settings: Mock GeneralSettings fixture.
        """
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        with (
            patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls,
            patch(
                "dlkit.interfaces.cli.commands.convert.load_config", return_value=sample_settings
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

            # Verify config was loaded
            mock_load.assert_called_once_with(sample_config_file)

            # Verify command was called with settings
            call_args = mock_cmd.execute.call_args
            input_data = call_args[0][0]
            settings = call_args[0][1]

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
        """Test conversion with config and batch size for validation.

        Args:
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
            sample_convert_result: Mock ConvertResult fixture.
            sample_config_file: Sample configuration file fixture.
            sample_settings: Mock GeneralSettings fixture.
        """
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

            # Verify batch size was passed for validation
            call_args = mock_cmd.execute.call_args
            input_data = call_args[0][0]
            assert input_data.batch_size == 8

    def test_convert_with_custom_opset(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        """Test conversion with custom ONNX opset version.

        Args:
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
            sample_convert_result: Mock ConvertResult fixture.
        """
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        # Create a new result with custom opset
        from dlkit.interfaces.api.commands.convert_command import ConvertResult

        custom_result = ConvertResult(
            output_path=sample_convert_result.output_path,
            opset=11,
            inputs=sample_convert_result.inputs,
        )

        with patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls:
            mock_cmd = Mock()
            mock_cmd.execute.return_value = custom_result
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
                    "--opset",
                    "11",
                ],
            )

            assert result.exit_code == 0
            assert "Opset: 11" in result.stdout

            # Verify opset was passed correctly
            call_args = mock_cmd.execute.call_args
            input_data = call_args[0][0]
            assert input_data.opset == 11

    def test_convert_with_multiple_inputs(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        """Test conversion with multiple input shapes.

        Args:
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
            sample_convert_result: Mock ConvertResult fixture.
        """
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        # Create a new result with multiple inputs
        from dlkit.interfaces.api.commands.convert_command import ConvertResult

        multi_input_result = ConvertResult(
            output_path=sample_convert_result.output_path,
            opset=sample_convert_result.opset,
            inputs=[(1, 3, 224, 224), (1, 10)],
        )

        with patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls:
            mock_cmd = Mock()
            mock_cmd.execute.return_value = multi_input_result
            mock_cmd_cls.return_value = mock_cmd

            result = cli_runner.invoke(
                cli_app,
                [
                    "convert",
                    "entry",
                    str(checkpoint_path),
                    str(output_path),
                    "--shape",
                    "3,224,224;10",
                ],
            )

            assert result.exit_code == 0
            assert "(1, 3, 224, 224), (1, 10)" in result.stdout

            # Verify shape was parsed correctly
            call_args = mock_cmd.execute.call_args
            input_data = call_args[0][0]
            assert input_data.shape == "3,224,224;10"

    def test_convert_short_options(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
        sample_config_file: Path,
        sample_settings: Mock,
    ) -> None:
        """Test conversion with short CLI option flags.

        Args:
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
            sample_convert_result: Mock ConvertResult fixture.
            sample_config_file: Sample configuration file fixture.
            sample_settings: Mock GeneralSettings fixture.
        """
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

            # Verify parameters were parsed correctly
            call_args = mock_cmd.execute.call_args
            input_data = call_args[0][0]
            assert input_data.batch_size == 16


class TestConvertCommandErrors:
    """Test error handling scenarios for convert command."""

    def test_missing_both_shape_and_config(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test error when neither shape nor config is provided.

        Args:
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
        """
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        result = cli_runner.invoke(
            cli_app, ["convert", "entry", str(checkpoint_path), str(output_path)]
        )

        assert result.exit_code == 1
        assert "Provide either --shape" in result.stdout

    def test_conversion_command_failure(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test handling of ConvertCommand execution failure.

        Args:
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
        """
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        error_msg = "Checkpoint file not found"

        with patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls:
            mock_cmd = Mock()
            mock_cmd.execute.side_effect = WorkflowError(error_msg, {"command": "convert"})
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

            assert result.exit_code == 1
            assert "Export failed:" in result.stdout
            assert error_msg in result.stdout

    def test_config_loading_failure(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test handling of config loading failure.

        Args:
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
        """
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"
        config_path = tmp_path / "invalid_config.toml"
        config_path.write_text("invalid toml content [")

        error_msg = "Invalid configuration format"

        with patch("dlkit.interfaces.cli.commands.convert.load_config") as mock_load:
            mock_load.side_effect = Exception(error_msg)

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
            assert "Export failed:" in result.stdout
            assert error_msg in result.stdout

    def test_nonexistent_checkpoint_path(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test error when checkpoint file doesn't exist.

        Args:
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
        """
        nonexistent_checkpoint = tmp_path / "nonexistent.ckpt"
        output_path = tmp_path / "model.onnx"

        with patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls:
            mock_cmd = Mock()
            mock_cmd.execute.side_effect = WorkflowError(
                f"Checkpoint file not found: {nonexistent_checkpoint}",
                {"command": "convert", "checkpoint": str(nonexistent_checkpoint)},
            )
            mock_cmd_cls.return_value = mock_cmd

            result = cli_runner.invoke(
                cli_app,
                [
                    "convert",
                    "entry",
                    str(nonexistent_checkpoint),
                    str(output_path),
                    "--shape",
                    "3,224,224",
                ],
            )

            assert result.exit_code == 1
            assert "Export failed:" in result.stdout
            assert "Checkpoint file not found" in result.stdout

    def test_generic_exception_handling(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test handling of generic unexpected exceptions.

        Args:
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
        """
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        error_msg = "Unexpected system error"

        with patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls:
            mock_cmd = Mock()
            mock_cmd.execute.side_effect = RuntimeError(error_msg)
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

            assert result.exit_code == 1
            assert "Export failed:" in result.stdout
            assert error_msg in result.stdout


class TestConvertCommandParameterValidation:
    """Test CLI parameter parsing and validation."""

    @pytest.mark.parametrize(
        "scenario_name",
        [
            "shape_basic",
            "shape_with_batch",
            "config_basic",
            "config_with_batch",
            "custom_opset",
            "multiple_inputs",
        ],
    )
    def test_parameter_scenarios(
        self,
        scenario_name: str,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
        convert_cli_scenarios: dict[str, dict[str, Any]],
        sample_config_file: Path,
        sample_settings: Mock,
    ) -> None:
        """Test various CLI parameter combinations using scenarios.

        Args:
            scenario_name: Name of the test scenario.
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
            sample_convert_result: Mock ConvertResult fixture.
            convert_cli_scenarios: CLI scenarios fixture.
            sample_config_file: Sample configuration file fixture.
            sample_settings: Mock GeneralSettings fixture.
        """
        scenario = convert_cli_scenarios[scenario_name]

        # Create required files
        checkpoint_path = tmp_path / "checkpoint.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "output.onnx"

        # Replace placeholders in args
        args = []
        for arg in scenario["args"]:
            if arg == "checkpoint.ckpt":
                args.append(str(checkpoint_path))
            elif arg == "output.onnx":
                args.append(str(output_path))
            elif arg == "config.toml":
                args.append(str(sample_config_file))
            else:
                args.append(arg)

        if "config.toml" in scenario["args"]:
            with (
                patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls,
                patch(
                    "dlkit.interfaces.cli.commands.convert.load_config",
                    return_value=sample_settings,
                ),
            ):
                mock_cmd = Mock()
                mock_cmd.execute.return_value = sample_convert_result
                mock_cmd_cls.return_value = mock_cmd

                result = cli_runner.invoke(cli_app, ["convert", "entry"] + args)

                assert result.exit_code == 0

                # Verify command was called with expected parameters
                call_args = mock_cmd.execute.call_args
                input_data = call_args[0][0]

                assert input_data.shape == scenario["expected_shape"]
                assert input_data.batch_size == scenario["expected_batch_size"]
                assert input_data.opset == scenario["expected_opset"]
        else:
            with patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls:
                mock_cmd = Mock()
                mock_cmd.execute.return_value = sample_convert_result
                mock_cmd_cls.return_value = mock_cmd

                result = cli_runner.invoke(cli_app, ["convert", "entry"] + args)

                assert result.exit_code == 0

                # Verify command was called with expected parameters
                call_args = mock_cmd.execute.call_args
                input_data = call_args[0][0]

                assert input_data.shape == scenario["expected_shape"]
                assert input_data.batch_size == scenario["expected_batch_size"]
                assert input_data.opset == scenario["expected_opset"]

    def test_path_handling_with_spaces(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        """Test path handling with spaces in filenames.

        Args:
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
            sample_convert_result: Mock ConvertResult fixture.
        """
        # Create paths with spaces
        checkpoint_path = tmp_path / "model checkpoint.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "output model.onnx"

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

            # Verify paths were handled correctly
            call_args = mock_cmd.execute.call_args
            input_data = call_args[0][0]
            assert input_data.checkpoint_path == checkpoint_path
            assert input_data.output_path == output_path

    def test_console_output_formatting(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test Rich console output formatting for success case.

        Args:
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
        """
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        # Create a detailed convert result
        from dlkit.interfaces.api.commands.convert_command import ConvertResult

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

            # Check formatted output components
            assert "✅ Export successful" in result.stdout
            assert "Output: /path/to/exported/model.onnx" in result.stdout
            assert "Opset: 17" in result.stdout
            assert "Inputs: (1, 3, 224, 224), (1, 512)" in result.stdout
            assert "ONNX Export" in result.stdout  # Panel title
