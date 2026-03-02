"""Property-based tests for ConvertCommand using Hypothesis."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch
from typing import Any

import pytest
from hypothesis import given, strategies as st, assume, settings as hypothesis_settings, HealthCheck
import torch

from dlkit.interfaces.api.commands.convert_command import (
    ConvertCommand,
    ConvertCommandInput,
    ConvertResult,
)
from dlkit.interfaces.api.domain.errors import WorkflowError
from ._helpers import (
    create_shape_spec,
    extract_batch_dimensions,
    validate_tensor_shape_consistency,
    create_expected_input_names,
)


# Hypothesis strategies for generating test dataflow
positive_dimensions = st.integers(min_value=1, max_value=1000)
valid_batch_sizes = st.integers(min_value=1, max_value=64)
valid_opset_versions = st.integers(min_value=9, max_value=20)
invalid_opset_versions = st.integers(max_value=8)

# Shape generation strategies
single_dimension = st.lists(positive_dimensions, min_size=1, max_size=4)
multiple_dimensions = st.lists(
    st.lists(positive_dimensions, min_size=1, max_size=4), min_size=1, max_size=3
)

# String-based shape specifications
shape_spec_single = single_dimension.map(create_shape_spec)
shape_spec_multi = multiple_dimensions.map(
    lambda dims_list: ";".join(create_shape_spec(dims) for dims in dims_list)
)

# File path strategies
valid_file_extensions = st.sampled_from([".onnx", ".ONNX", ".proto"])
checkpoint_extensions = st.sampled_from([".ckpt", ".pth", ".pt"])


class TestConvertCommandProperties:
    """Property-based tests for ConvertCommand behavior."""

    @given(batch_size=valid_batch_sizes, opset=valid_opset_versions, shape_dims=single_dimension)
    @hypothesis_settings(
        max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_validate_input_accepts_valid_parameters(
        self,
        batch_size: int,
        opset: int,
        shape_dims: list[int],
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Property: validate_input should accept all valid parameter combinations.

        Args:
            batch_size: Valid batch size
            opset: Valid ONNX opset version
            shape_dims: Valid shape dimensions
        """
        tmp_path = tmp_path_factory.mktemp("convert_valid")

        # Setup files
        checkpoint = tmp_path / "model.ckpt"
        checkpoint.write_text("mock")
        output = tmp_path / "model.onnx"

        input_data = ConvertCommandInput(
            checkpoint_path=checkpoint,
            output_path=output,
            shape=create_shape_spec(shape_dims),
            batch_size=batch_size,
            opset=opset,
        )

        command = ConvertCommand()

        # Should not raise any exception
        command.validate_input(input_data, Mock())

    @given(opset=invalid_opset_versions)
    @hypothesis_settings(
        max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_validate_input_rejects_invalid_opset(
        self,
        opset: int,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Property: validate_input should reject opset versions < 9.

        Args:
            opset: Invalid ONNX opset version
        """
        tmp_path = tmp_path_factory.mktemp("convert_invalid")

        # Setup files
        checkpoint = tmp_path / "model.ckpt"
        checkpoint.write_text("mock")
        output = tmp_path / "model.onnx"

        input_data = ConvertCommandInput(
            checkpoint_path=checkpoint,
            output_path=output,
            shape="3,224,224",
            batch_size=4,
            opset=opset,
        )

        command = ConvertCommand()

        with pytest.raises(WorkflowError) as exc_info:
            command.validate_input(input_data, Mock())

        assert "Unsupported opset version" in str(exc_info.value)
        assert exc_info.value.context["opset"] == opset

    @given(batch_size=valid_batch_sizes, shape_dims=single_dimension)
    @hypothesis_settings(max_examples=20, deadline=5000)
    def test_parse_shapes_produces_consistent_batch_dimensions(
        self, batch_size: int, shape_dims: list[int]
    ) -> None:
        """Property: _parse_or_infer_shapes should produce consistent batch dimensions.

        Args:
            batch_size: Batch size for shape parsing
            shape_dims: Shape dimensions to parse
        """
        command = ConvertCommand()
        shape_spec = create_shape_spec(shape_dims)

        shapes, inferred = command._parse_or_infer_shapes(
            shape_spec=shape_spec, settings=Mock(), default_batch=batch_size
        )

        # Property: all shapes should have the same batch dimension
        assert validate_tensor_shape_consistency(shapes)
        batch_dims = extract_batch_dimensions(shapes)
        assert len(batch_dims) == 1
        assert next(iter(batch_dims)) == batch_size
        assert not inferred  # Should not be inferred when explicit shape is provided

    @given(shape_dims_list=multiple_dimensions, batch_size=valid_batch_sizes)
    @hypothesis_settings(max_examples=15, deadline=5000)
    def test_parse_multiple_shapes_maintains_consistency(
        self, shape_dims_list: list[list[int]], batch_size: int
    ) -> None:
        """Property: Multiple input shapes should maintain batch consistency.

        Args:
            shape_dims_list: List of shape dimension lists
            batch_size: Batch size for all shapes
        """
        assume(len(shape_dims_list) >= 2)  # Only test multi-input cases

        command = ConvertCommand()
        shape_spec = ";".join(create_shape_spec(dims) for dims in shape_dims_list)

        shapes, inferred = command._parse_or_infer_shapes(
            shape_spec=shape_spec, settings=Mock(), default_batch=batch_size
        )

        # Properties
        assert len(shapes) == len(shape_dims_list)
        assert validate_tensor_shape_consistency(shapes)
        assert not inferred

        # All shapes should have the expected structure
        for i, shape in enumerate(shapes):
            expected_shape = tuple([batch_size] + shape_dims_list[i])
            assert shape == expected_shape

    @given(shape_dims=single_dimension, batch_size=valid_batch_sizes)
    @hypothesis_settings(max_examples=15, deadline=5000)
    def test_shape_parsing_handles_x_separator(
        self, shape_dims: list[int], batch_size: int
    ) -> None:
        """Property: Shape parsing should handle both comma and 'x' separators.

        Args:
            shape_dims: Shape dimensions
            batch_size: Batch size
        """
        command = ConvertCommand()

        # Test both comma and 'x' separators
        comma_spec = create_shape_spec(shape_dims)
        x_spec = "x".join(str(d) for d in shape_dims)

        comma_shapes, _ = command._parse_or_infer_shapes(
            shape_spec=comma_spec, settings=Mock(), default_batch=batch_size
        )

        x_shapes, _ = command._parse_or_infer_shapes(
            shape_spec=x_spec, settings=Mock(), default_batch=batch_size
        )

        # Property: both separators should produce identical results
        assert comma_shapes == x_shapes
        expected_shape = tuple([batch_size] + shape_dims)
        assert comma_shapes == [expected_shape]

    @given(num_inputs=st.integers(min_value=1, max_value=5))
    @hypothesis_settings(max_examples=10, deadline=5000)
    def test_input_names_generation_consistency(self, num_inputs: int) -> None:
        """Property: Input name generation should be consistent and unique.

        Args:
            num_inputs: Number of inputs to generate names for
        """
        names = create_expected_input_names(num_inputs)

        # Properties
        assert len(names) == num_inputs
        assert len(set(names)) == num_inputs  # All names should be unique

        if num_inputs == 1:
            assert names == ["input"]
        else:
            for i, name in enumerate(names):
                assert name == f"input{i}"

    @given(
        dims_with_invalid=st.lists(
            st.one_of(
                st.integers(max_value=0),  # Invalid: zero or negative
                st.text().filter(
                    lambda x: not x.isdigit() and x.strip() != ""
                ),  # Invalid: non-numeric
            ),
            min_size=1,
            max_size=3,
        )
    )
    @hypothesis_settings(max_examples=10, deadline=5000)
    def test_invalid_shape_specs_raise_workflow_error(self, dims_with_invalid: list[Any]) -> None:
        """Property: Invalid shape specifications should raise WorkflowError.

        Args:
            dims_with_invalid: List containing invalid dimension values
        """
        command = ConvertCommand()

        # Create shape spec with invalid dimensions
        invalid_spec = ",".join(str(d) for d in dims_with_invalid)

        with pytest.raises(WorkflowError):
            command._parse_or_infer_shapes(
                shape_spec=invalid_spec, settings=Mock(), default_batch=1
            )

    @given(batch_size=valid_batch_sizes, opset=valid_opset_versions)
    @hypothesis_settings(
        max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_execute_result_properties(
        self,
        batch_size: int,
        opset: int,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Property: execute should return ConvertResult with consistent properties.

        Args:
            batch_size: Batch size for execution
            opset: ONNX opset version
        """
        tmp_path = tmp_path_factory.mktemp("convert_execute")

        with (
            patch(
                "dlkit.interfaces.api.commands.convert_command.WrapperFactory.create_wrapper_from_checkpoint"
            ) as mock_wrapper_factory,
            patch("dlkit.interfaces.api.commands.convert_command.torch.ones") as mock_torch_ones,
            patch(
                "dlkit.interfaces.api.commands.convert_command.torch.onnx.export"
            ) as mock_torch_export,
        ):
            # Setup files and mocks
            checkpoint = tmp_path / "model.ckpt"
            checkpoint.write_text("mock")
            output = tmp_path / "model.onnx"

            mock_wrapper = Mock()
            mock_wrapper_factory.return_value = mock_wrapper
            mock_torch_ones.return_value = Mock(spec=torch.Tensor)

            input_data = ConvertCommandInput(
                checkpoint_path=checkpoint,
                output_path=output,
                shape="3,224,224",
                batch_size=batch_size,
                opset=opset,
            )

            command = ConvertCommand()
            result = command.execute(input_data, Mock())

            # Properties
            assert isinstance(result, ConvertResult)
            assert result.output_path == output
            assert result.opset == opset
            assert len(result.inputs) >= 1
            assert validate_tensor_shape_consistency(result.inputs)

            # Verify batch dimension consistency
            batch_dims = extract_batch_dimensions(result.inputs)
            assert len(batch_dims) == 1
            assert next(iter(batch_dims)) == batch_size

    @given(
        checkpoint_name=st.text(min_size=1, max_size=20).filter(lambda x: x.strip() != ""),
        output_name=st.text(min_size=1, max_size=20).filter(lambda x: x.strip() != ""),
    )
    @hypothesis_settings(
        max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_path_handling_properties(
        self,
        checkpoint_name: str,
        output_name: str,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Property: Path handling should work with various valid filenames.

        Args:
            checkpoint_name: Generated checkpoint filename
            output_name: Generated output filename
        """
        tmp_path = tmp_path_factory.mktemp("convert_paths")

        # Clean up names to ensure they're valid filenames
        safe_checkpoint = "".join(c for c in checkpoint_name if c.isalnum() or c in "._-")
        safe_output = "".join(c for c in output_name if c.isalnum() or c in "._-")

        if not safe_checkpoint:
            safe_checkpoint = "checkpoint"
        if not safe_output:
            safe_output = "output"

        checkpoint = tmp_path / f"{safe_checkpoint}.ckpt"
        output = tmp_path / f"{safe_output}.onnx"
        checkpoint.write_text("mock")

        input_data = ConvertCommandInput(
            checkpoint_path=checkpoint,
            output_path=output,
            shape="3,224,224",
            batch_size=4,
            opset=17,
        )

        command = ConvertCommand()

        # Should not raise exception for valid paths
        command.validate_input(input_data, Mock())

    @given(batch_size_1=valid_batch_sizes, batch_size_2=valid_batch_sizes)
    @hypothesis_settings(
        max_examples=15, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_batch_size_mismatch_detection(
        self,
        batch_size_1: int,
        batch_size_2: int,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Property: Batch size mismatches should be consistently detected.

        Args:
            batch_size_1: First batch size (from CLI)
            batch_size_2: Second batch size (from inferred shapes)
        """
        assume(batch_size_1 != batch_size_2)  # Only test mismatch cases

        tmp_path = tmp_path_factory.mktemp("convert_batch_mismatch")

        checkpoint = tmp_path / "model.ckpt"
        checkpoint.write_text("mock")
        output = tmp_path / "model.onnx"

        input_data = ConvertCommandInput(
            checkpoint_path=checkpoint,
            output_path=output,
            shape=None,  # Force inference mode
            batch_size=batch_size_1,
            opset=17,
        )

        command = ConvertCommand()

        # Mock _parse_or_infer_shapes to return shape with different batch size
        inferred_shapes = [(batch_size_2, 3, 224, 224)]

        with patch.object(command, "_parse_or_infer_shapes", return_value=(inferred_shapes, True)):
            with patch(
                "dlkit.interfaces.api.commands.convert_command.WrapperFactory.create_wrapper_from_checkpoint"
            ):
                with pytest.raises(WorkflowError) as exc_info:
                    command.execute(input_data, Mock())

                # Properties of the error
                assert "Batch size mismatch" in str(exc_info.value)
                assert exc_info.value.context["expected_batch"] == batch_size_1
                assert batch_size_2 in exc_info.value.context["found_batches"]
