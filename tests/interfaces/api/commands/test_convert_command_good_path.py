"""Test ConvertCommand good path scenarios."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from dlkit.interfaces.api.commands.convert_command import (
    ConvertCommand,
    ConvertCommandInput,
    ConvertResult,
)
from ._helpers import (
    create_multi_input_shape_spec,
    create_expected_input_names,
    validate_dynamic_axes_structure,
)


def test_convert_command_initialization() -> None:
    """Test ConvertCommand initializes correctly."""
    command = ConvertCommand()
    assert command.command_name == "convert"

    custom_command = ConvertCommand("custom_convert")
    assert custom_command.command_name == "custom_convert"


def test_convert_command_input_dataclass_creation(
    mock_checkpoint_path: Path, mock_output_path: Path
) -> None:
    """Test ConvertCommandInput dataclass creation with all parameters."""
    input_data = ConvertCommandInput(
        checkpoint_path=mock_checkpoint_path,
        output_path=mock_output_path,
        shape="3,224,224",
        batch_size=4,
        opset=17,
    )

    assert input_data.checkpoint_path == mock_checkpoint_path
    assert input_data.output_path == mock_output_path
    assert input_data.shape == "3,224,224"
    assert input_data.batch_size == 4
    assert input_data.opset == 17


def test_convert_command_input_defaults() -> None:
    """Test ConvertCommandInput dataclass with default values."""
    input_data = ConvertCommandInput(checkpoint_path="checkpoint.ckpt", output_path="output.onnx")

    assert input_data.shape is None
    assert input_data.batch_size is None
    assert input_data.opset == 17


def test_convert_result_dataclass_creation(mock_output_path: Path) -> None:
    """Test ConvertResult dataclass creation."""
    result = ConvertResult(output_path=mock_output_path, opset=17, inputs=[(4, 3, 224, 224)])

    assert result.output_path == mock_output_path
    assert result.opset == 17
    assert result.inputs == [(4, 3, 224, 224)]


def test_validate_input_with_valid_data(
    valid_convert_input: ConvertCommandInput, mock_settings: Mock
) -> None:
    """Test validate_input with valid input"""
    command = ConvertCommand()

    # Should not raise any exception
    command.validate_input(valid_convert_input, mock_settings)


def test_validate_input_with_shape_and_batch_size(
    mock_checkpoint_path: Path, mock_output_path: Path, mock_settings: Mock
) -> None:
    """Test validate_input with various valid shape and batch size combinations."""
    command = ConvertCommand()

    test_cases = [
        {"shape": "28,28", "batch_size": 1},
        {"shape": "3,224,224", "batch_size": 4},
        {"shape": "100", "batch_size": 8},
        {"shape": "10,20,30", "batch_size": 16},
    ]

    for case in test_cases:
        input_data = ConvertCommandInput(
            checkpoint_path=mock_checkpoint_path,
            output_path=mock_output_path,
            shape=case["shape"],
            batch_size=case["batch_size"],
            opset=17,
        )

        # Should not raise any exception
        command.validate_input(input_data, mock_settings)


def test_validate_input_without_shape_specification(
    valid_convert_input_no_shape: ConvertCommandInput, mock_settings: Mock
) -> None:
    """Test validate_input without shape specification (config inference mode)."""
    command = ConvertCommand()

    # Should not raise any exception when no shape is specified
    command.validate_input(valid_convert_input_no_shape, mock_settings)


def test_parse_or_infer_shapes_with_single_shape_spec(mock_settings: Mock) -> None:
    """Test _parse_or_infer_shapes with single shape specification."""
    command = ConvertCommand()

    shapes, inferred_from_cfg = command._parse_or_infer_shapes(
        shape_spec="3,224,224", settings=mock_settings, default_batch=4
    )

    assert not inferred_from_cfg
    assert shapes == [(4, 3, 224, 224)]


def test_parse_or_infer_shapes_with_multiple_shape_specs(mock_settings: Mock) -> None:
    """Test _parse_or_infer_shapes with multiple shape specifications."""
    command = ConvertCommand()

    shape_spec = create_multi_input_shape_spec([[3, 224, 224], [100]])

    shapes, inferred_from_cfg = command._parse_or_infer_shapes(
        shape_spec=shape_spec, settings=mock_settings, default_batch=2
    )

    assert not inferred_from_cfg
    assert shapes == [(2, 3, 224, 224), (2, 100)]


def test_parse_or_infer_shapes_with_x_separator(mock_settings: Mock) -> None:
    """Test _parse_or_infer_shapes with 'x' as dimension separator."""
    command = ConvertCommand()

    shapes, inferred_from_cfg = command._parse_or_infer_shapes(
        shape_spec="3x224x224", settings=mock_settings, default_batch=1
    )

    assert not inferred_from_cfg
    assert shapes == [(1, 3, 224, 224)]


@patch("dlkit.interfaces.api.commands.convert_command.FlexibleBuildStrategy")
def test_parse_or_infer_shapes_from_dataloader_dict_batch(
    mock_strategy_class: Mock,
    mock_build_strategy: Mock,
    mock_build_components: Mock,
    mock_dataloader: Mock,
    mock_dataloader_batch_dict: dict,
    mock_settings: Mock,
) -> None:
    """Test _parse_or_infer_shapes with dataloader inference (dict batch)."""
    mock_strategy_class.return_value = mock_build_strategy
    command = ConvertCommand()

    shapes, inferred_from_cfg = command._parse_or_infer_shapes(
        shape_spec=None, settings=mock_settings, default_batch=4
    )

    assert inferred_from_cfg
    assert shapes == [(4, 3, 224, 224)]
    mock_strategy_class.assert_called_once()
    mock_build_strategy.build.assert_called_once_with(mock_settings)


@patch("dlkit.interfaces.api.commands.convert_command.FlexibleBuildStrategy")
def test_parse_or_infer_shapes_from_dataloader_tuple_batch(
    mock_strategy_class: Mock,
    mock_build_strategy: Mock,
    mock_build_components: Mock,
    mock_dataloader_batch_tuple: tuple,
    mock_settings: Mock,
) -> None:
    """Test _parse_or_infer_shapes with dataloader inference (tuple batch)."""
    # Set up dataloader to return tuple batch
    mock_dataloader = Mock()
    mock_dataloader.__iter__ = Mock(return_value=iter([mock_dataloader_batch_tuple]))

    mock_datamodule = Mock()
    mock_datamodule.predict_dataloader = Mock(return_value=mock_dataloader)
    mock_datamodule.val_dataloader = Mock(return_value=None)
    mock_datamodule.test_dataloader = Mock(return_value=None)
    mock_datamodule.train_dataloader = Mock(return_value=None)

    mock_build_components.datamodule = mock_datamodule
    mock_build_strategy.build = Mock(return_value=mock_build_components)
    mock_strategy_class.return_value = mock_build_strategy

    command = ConvertCommand()

    shapes, inferred_from_cfg = command._parse_or_infer_shapes(
        shape_spec=None, settings=mock_settings, default_batch=4
    )

    assert inferred_from_cfg
    assert shapes == [(4, 3, 224, 224)]


@patch("dlkit.interfaces.api.commands.convert_command.FlexibleBuildStrategy")
def test_parse_or_infer_shapes_from_dataloader_tensor_batch(
    mock_strategy_class: Mock,
    mock_build_strategy: Mock,
    mock_build_components: Mock,
    mock_dataloader_batch_tensor: Mock,
    mock_settings: Mock,
) -> None:
    """Test _parse_or_infer_shapes with dataloader inference (single tensor batch)."""
    # Set up dataloader to return single tensor batch
    mock_dataloader = Mock()
    mock_dataloader.__iter__ = Mock(return_value=iter([mock_dataloader_batch_tensor]))

    mock_datamodule = Mock()
    mock_datamodule.predict_dataloader = Mock(return_value=mock_dataloader)
    mock_datamodule.val_dataloader = Mock(return_value=None)
    mock_datamodule.test_dataloader = Mock(return_value=None)
    mock_datamodule.train_dataloader = Mock(return_value=None)

    mock_build_components.datamodule = mock_datamodule
    mock_build_strategy.build = Mock(return_value=mock_build_components)
    mock_strategy_class.return_value = mock_build_strategy

    command = ConvertCommand()

    shapes, inferred_from_cfg = command._parse_or_infer_shapes(
        shape_spec=None, settings=mock_settings, default_batch=4
    )

    assert inferred_from_cfg
    assert shapes == [(4, 3, 224, 224)]


@patch("dlkit.interfaces.api.commands.convert_command.torch.onnx.export")
@patch("dlkit.interfaces.api.commands.convert_command.torch.ones")
@patch(
    "dlkit.interfaces.api.commands.convert_command.WrapperFactory.create_wrapper_from_checkpoint"
)
def test_execute_with_single_input_shape(
    mock_wrapper_factory: Mock,
    mock_torch_ones: Mock,
    mock_torch_export: Mock,
    valid_convert_input: ConvertCommandInput,
    mock_wrapper: Mock,
    mock_settings: Mock,
) -> None:
    """Test execute with single input shape specification."""
    # Setup mocks
    mock_wrapper_factory.return_value = mock_wrapper
    mock_example_tensor = Mock(spec=torch.Tensor)
    mock_torch_ones.return_value = mock_example_tensor

    command = ConvertCommand()

    result = command.execute(valid_convert_input, mock_settings)

    # Verify result
    assert result.output_path == valid_convert_input.output_path
    assert result.opset == valid_convert_input.opset
    assert result.inputs == [(4, 3, 224, 224)]

    # Verify wrapper loading and evaluation
    mock_wrapper_factory.assert_called_once_with(str(valid_convert_input.checkpoint_path))
    mock_wrapper.eval.assert_called_once()

    # Verify tensor creation
    mock_torch_ones.assert_called_once_with((4, 3, 224, 224), dtype=torch.float32)

    # Verify ONNX export
    expected_dynamic_axes = validate_dynamic_axes_structure(["input"])
    mock_torch_export.assert_called_once_with(
        mock_wrapper,
        mock_example_tensor,
        str(valid_convert_input.output_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=expected_dynamic_axes,
    )


@patch("dlkit.interfaces.api.commands.convert_command.torch.onnx.export")
@patch("dlkit.interfaces.api.commands.convert_command.torch.ones")
@patch(
    "dlkit.interfaces.api.commands.convert_command.WrapperFactory.create_wrapper_from_checkpoint"
)
def test_execute_with_multiple_input_shapes(
    mock_wrapper_factory: Mock,
    mock_torch_ones: Mock,
    mock_torch_export: Mock,
    mock_checkpoint_path: Path,
    mock_output_path: Path,
    mock_wrapper: Mock,
    mock_settings: Mock,
) -> None:
    """Test execute with multiple input shape specifications."""
    # Setup input with multiple shapes
    multi_shape_spec = create_multi_input_shape_spec([[3, 224, 224], [100]])
    input_data = ConvertCommandInput(
        checkpoint_path=mock_checkpoint_path,
        output_path=mock_output_path,
        shape=multi_shape_spec,
        batch_size=2,
        opset=17,
    )

    # Setup mocks
    mock_wrapper_factory.return_value = mock_wrapper
    mock_tensor1 = Mock(spec=torch.Tensor)
    mock_tensor2 = Mock(spec=torch.Tensor)
    mock_torch_ones.side_effect = [mock_tensor1, mock_tensor2]

    command = ConvertCommand()

    result = command.execute(input_data, mock_settings)

    # Verify result
    assert result.output_path == input_data.output_path
    assert result.opset == input_data.opset
    assert result.inputs == [(2, 3, 224, 224), (2, 100)]

    # Verify tensor creation for multiple inputs
    expected_calls = [((2, 3, 224, 224), torch.float32), ((2, 100), torch.float32)]
    for i, (shape, dtype) in enumerate(expected_calls):
        assert mock_torch_ones.call_args_list[i][0] == (shape,)
        assert mock_torch_ones.call_args_list[i][1] == {"dtype": dtype}

    # Verify ONNX export with multiple inputs
    expected_input_names = create_expected_input_names(2)
    expected_dynamic_axes = validate_dynamic_axes_structure(expected_input_names)
    mock_torch_export.assert_called_once_with(
        mock_wrapper,
        (mock_tensor1, mock_tensor2),
        str(input_data.output_path),
        opset_version=17,
        input_names=expected_input_names,
        output_names=["output"],
        dynamic_axes=expected_dynamic_axes,
    )


@patch("dlkit.interfaces.api.commands.convert_command.torch.onnx.export")
@patch("dlkit.interfaces.api.commands.convert_command.torch.ones")
@patch(
    "dlkit.interfaces.api.commands.convert_command.WrapperFactory.create_wrapper_from_checkpoint"
)
@patch("dlkit.interfaces.api.commands.convert_command.FlexibleBuildStrategy")
def test_execute_with_config_based_shape_inference(
    mock_strategy_class: Mock,
    mock_wrapper_factory: Mock,
    mock_torch_ones: Mock,
    mock_torch_export: Mock,
    valid_convert_input_no_shape: ConvertCommandInput,
    mock_wrapper: Mock,
    mock_build_strategy: Mock,
    mock_build_components: Mock,
    mock_dataloader: Mock,
    mock_settings: Mock,
) -> None:
    """Test execute with config-based shape inference."""
    # Setup mocks for shape inference
    mock_strategy_class.return_value = mock_build_strategy
    mock_wrapper_factory.return_value = mock_wrapper
    mock_example_tensor = Mock(spec=torch.Tensor)
    mock_torch_ones.return_value = mock_example_tensor

    command = ConvertCommand()

    result = command.execute(valid_convert_input_no_shape, mock_settings)

    # Verify result (should use inferred shape from dataloader)
    assert result.output_path == valid_convert_input_no_shape.output_path
    assert result.opset == valid_convert_input_no_shape.opset
    assert result.inputs == [(4, 3, 224, 224)]  # From mock dataloader batch

    # Verify shape inference was triggered
    mock_strategy_class.assert_called_once()
    mock_build_strategy.build.assert_called_once_with(mock_settings)


def test_execute_validates_input_before_processing(
    mock_checkpoint_path: Path, tmp_path: Path, mock_settings: Mock
) -> None:
    """Test that execute calls validate_input before processing."""
    # Create input with directory as output path (invalid)
    output_dir = tmp_path / "output_dir"
    output_dir.mkdir()

    input_data = ConvertCommandInput(
        checkpoint_path=mock_checkpoint_path,
        output_path=output_dir,  # This should fail validation
        shape="3,224,224",
        batch_size=4,
        opset=17,
    )

    command = ConvertCommand()

    with pytest.raises(Exception):  # Should raise WorkflowError due to validation
        command.execute(input_data, mock_settings)
