"""Property-based tests for CLI convert command using Hypothesis.

This module provides property-based testing for convert command parameters
to ensure robust validation across a wide range of inputs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from hypothesis import given, strategies as st, settings, HealthCheck
from typer.testing import CliRunner

from dlkit.interfaces.cli.app import app as cli_app

# Constants for property boundaries
MIN_OPSET = 9
MAX_OPSET = 20
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 512
MIN_DIMENSION = 1
MAX_DIMENSION = 2048
MAX_SHAPE_PARTS = 5


class TestConvertCommandProperties:
    """Property-based tests for convert command parameter validation."""

    @given(
        shape_dims=st.lists(
            st.integers(min_value=MIN_DIMENSION, max_value=MAX_DIMENSION),
            min_size=1,
            max_size=MAX_SHAPE_PARTS,
        ),
        batch_size=st.integers(min_value=MIN_BATCH_SIZE, max_value=MAX_BATCH_SIZE),
        opset=st.integers(min_value=MIN_OPSET, max_value=MAX_OPSET),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_valid_shape_parameters_property(
        self,
        shape_dims: list[int],
        batch_size: int,
        opset: int,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        """Test that valid shape parameters are processed correctly.

        Args:
            shape_dims: List of shape dimensions to test.
            batch_size: Batch size to test.
            opset: ONNX opset version to test.
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
            sample_convert_result: Mock ConvertResult fixture.
        """
        # Create required files
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        # Create shape string from dimensions
        shape_str = ",".join(str(d) for d in shape_dims)

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

            # Valid parameters should always succeed
            assert result.exit_code == 0

            # Verify parameters were passed correctly
            call_args = mock_cmd.execute.call_args
            input_data = call_args[0][0]

            assert input_data.shape == shape_str
            assert input_data.batch_size == batch_size
            assert input_data.opset == opset

    @given(
        filename_base=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
            min_size=1,
            max_size=50,
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_valid_filename_handling_property(
        self,
        filename_base: str,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        """Test that valid filenames are handled correctly.

        Args:
            filename_base: Base filename to test.
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
            sample_convert_result: Mock ConvertResult fixture.
        """
        # Ensure valid filename by sanitizing
        safe_filename = "".join(c for c in filename_base if c.isalnum() or c in "_-")
        if not safe_filename:
            safe_filename = "model"  # Fallback for empty strings

        checkpoint_path = tmp_path / f"{safe_filename}.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / f"{safe_filename}_output.onnx"

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

    @given(
        num_inputs=st.integers(min_value=1, max_value=3),
        shape_dims_per_input=st.lists(
            st.lists(st.integers(min_value=MIN_DIMENSION, max_value=256), min_size=1, max_size=4),
            min_size=1,
            max_size=3,
        ),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_multiple_input_shapes_property(
        self,
        num_inputs: int,
        shape_dims_per_input: list[list[int]],
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        """Test multiple input shapes with property-based generation.

        Args:
            num_inputs: Number of input shapes to generate.
            shape_dims_per_input: List of dimension lists for each input.
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
            sample_convert_result: Mock ConvertResult fixture.
        """
        # Limit to the specified number of inputs
        limited_shapes = shape_dims_per_input[:num_inputs]

        # Create shape string with multiple inputs
        shape_parts = []
        for dims in limited_shapes:
            shape_str = ",".join(str(d) for d in dims)
            shape_parts.append(shape_str)

        multi_shape = ";".join(shape_parts)

        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        # Create a new result with multiple inputs
        from dlkit.interfaces.api.commands.convert_command import ConvertResult

        multi_result = ConvertResult(
            output_path=sample_convert_result.output_path,
            opset=sample_convert_result.opset,
            inputs=[tuple([1] + dims) for dims in limited_shapes],
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

            # Verify shape was processed correctly
            call_args = mock_cmd.execute.call_args
            input_data = call_args[0][0]
            assert input_data.shape == multi_shape

    @given(opset=st.integers(min_value=MIN_OPSET, max_value=MAX_OPSET))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_opset_version_boundary_property(
        self,
        opset: int,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        """Test ONNX opset version boundaries with property-based testing.

        Args:
            opset: ONNX opset version to test.
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
            sample_convert_result: Mock ConvertResult fixture.
        """
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        # Create a new result with the tested opset
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

            # Verify opset was passed correctly
            call_args = mock_cmd.execute.call_args
            input_data = call_args[0][0]
            assert input_data.opset == opset

    @given(batch_size=st.integers(min_value=MIN_BATCH_SIZE, max_value=MAX_BATCH_SIZE))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_batch_size_boundary_property(
        self,
        batch_size: int,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        """Test batch size boundaries with property-based testing.

        Args:
            batch_size: Batch size to test.
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
                    "784",
                    "--batch-size",
                    str(batch_size),
                ],
            )

            assert result.exit_code == 0

            # Verify batch size was passed correctly
            call_args = mock_cmd.execute.call_args
            input_data = call_args[0][0]
            assert input_data.batch_size == batch_size


class TestConvertCommandInvalidProperties:
    """Property-based tests for invalid parameter handling."""

    @given(
        invalid_shape=st.one_of(
            st.just(""),  # Empty string
            st.just("0,224,224"),  # Zero dimension
            st.just("-1,224,224"),  # Negative dimension
            st.just("abc,224,224"),  # Non-numeric
            st.just("3,224,"),  # Trailing comma
            st.just(",224,224"),  # Leading comma
            st.just("3,,224"),  # Double comma
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_invalid_shape_handling_property(
        self,
        invalid_shape: str,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test handling of various invalid shape formats.

        Args:
            invalid_shape: Invalid shape string to test.
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
        """
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        # Mock ConvertCommand to raise validation error for invalid shapes
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

            # Invalid shapes should result in error
            assert result.exit_code == 1
            assert "Export failed:" in result.stdout

    @given(
        opset=st.integers(max_value=MIN_OPSET - 1)  # Below minimum supported
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_invalid_opset_property(
        self,
        opset: int,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test handling of invalid ONNX opset versions.

        Args:
            opset: Invalid opset version to test.
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
        """
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        # Mock ConvertCommand to raise validation error for invalid opset
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

            # Invalid opset should result in error
            assert result.exit_code == 1
            assert "Export failed:" in result.stdout


class TestConvertCommandRobustness:
    """Robustness tests for edge cases and unusual parameter combinations."""

    @given(path_length=st.integers(min_value=1, max_value=100))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_path_length_robustness(
        self,
        path_length: int,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        """Test robustness with various path lengths.

        Args:
            path_length: Length of filename to generate.
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
            sample_convert_result: Mock ConvertResult fixture.
        """
        # Create filename with specified length (using safe characters)
        filename_base = "a" * min(path_length, 50)  # Limit to reasonable length

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

    @given(data=st.data())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_parameter_combination_robustness(
        self,
        data: st.DataObject,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_convert_result: Mock,
    ) -> None:
        """Test robustness with various parameter combinations.

        Args:
            data: Hypothesis dataflow strategy for generating combinations.
            cli_runner: Typer CLI test runner fixture.
            tmp_path: Pytest temporary directory fixture.
            sample_convert_result: Mock ConvertResult fixture.
        """
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("dummy checkpoint")
        output_path = tmp_path / "model.onnx"

        # Generate valid parameter combinations
        use_shape = data.draw(st.booleans())

        if use_shape:
            # Generate valid shape
            dims = data.draw(
                st.lists(st.integers(min_value=1, max_value=512), min_size=1, max_size=4)
            )
            shape_str = ",".join(str(d) for d in dims)
            batch_size = data.draw(st.integers(min_value=1, max_value=64))
            opset = data.draw(st.integers(min_value=MIN_OPSET, max_value=MAX_OPSET))

            args = [
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
            ]
        else:
            # Use config mode
            config_path = tmp_path / "config.toml"
            config_path.write_text("[SESSION]\nname = 'test'")

            batch_size = data.draw(st.integers(min_value=1, max_value=64))
            opset = data.draw(st.integers(min_value=MIN_OPSET, max_value=MAX_OPSET))

            args = [
                "convert",
                "entry",
                str(checkpoint_path),
                str(output_path),
                "--config",
                str(config_path),
                "--batch-size",
                str(batch_size),
                "--opset",
                str(opset),
            ]

        if not use_shape:
            with (
                patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls,
                patch("dlkit.interfaces.cli.commands.convert.load_config", return_value=Mock()),
            ):
                mock_cmd = Mock()
                mock_cmd.execute.return_value = sample_convert_result
                mock_cmd_cls.return_value = mock_cmd

                result = cli_runner.invoke(cli_app, args)

                # All valid parameter combinations should succeed
                assert result.exit_code == 0
        else:
            with patch("dlkit.interfaces.cli.commands.convert.ConvertCommand") as mock_cmd_cls:
                mock_cmd = Mock()
                mock_cmd.execute.return_value = sample_convert_result
                mock_cmd_cls.return_value = mock_cmd

                result = cli_runner.invoke(cli_app, args)

                # All valid parameter combinations should succeed
                assert result.exit_code == 0
