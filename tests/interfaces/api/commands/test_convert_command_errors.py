"""Test ConvertCommand error scenarios and edge cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from dlkit.interfaces.api.commands.convert_command import ConvertCommand, ConvertCommandInput
from dlkit.interfaces.api.domain.errors import WorkflowError


def test_validate_input_nonexistent_checkpoint() -> None:
    """Test validate_input raises WorkflowError for nonexistent checkpoint."""
    command = ConvertCommand()

    input_data = ConvertCommandInput(
        checkpoint_path="/nonexistent/path/model.ckpt",
        output_path="output.onnx",
        shape="3,224,224",
        batch_size=4,
        opset=17,
    )

    with pytest.raises(WorkflowError) as exc_info:
        command.validate_input(input_data, cast(Any, None))

    assert "Checkpoint file not found" in str(exc_info.value)
    assert exc_info.value.context["command"] == "convert"
    assert "checkpoint" in exc_info.value.context


def test_validate_input_directory_as_output_path(
    mock_checkpoint_path: Path, tmp_path: Path
) -> None:
    """Test validate_input raises WorkflowError when output path is directory."""
    command = ConvertCommand()

    # Create a directory as output path
    output_dir = tmp_path / "output_dir"
    output_dir.mkdir()

    input_data = ConvertCommandInput(
        checkpoint_path=mock_checkpoint_path,
        output_path=output_dir,
        shape="3,224,224",
        batch_size=4,
        opset=17,
    )

    with pytest.raises(WorkflowError) as exc_info:
        command.validate_input(input_data, cast(Any, None))

    assert "Output path points to a directory" in str(exc_info.value)
    assert exc_info.value.context["command"] == "convert"
    assert "output" in exc_info.value.context


def test_validate_input_invalid_opset_versions(
    mock_checkpoint_path: Path, mock_output_path: Path
) -> None:
    """Test validate_input raises WorkflowError for invalid opset versions."""
    command = ConvertCommand()

    invalid_opsets = [1, 5, 8, -1, 0]

    for invalid_opset in invalid_opsets:
        input_data = ConvertCommandInput(
            checkpoint_path=mock_checkpoint_path,
            output_path=mock_output_path,
            shape="3,224,224",
            batch_size=4,
            opset=invalid_opset,
        )

        with pytest.raises(WorkflowError) as exc_info:
            command.validate_input(input_data, cast(Any, None))

        assert "Unsupported opset version" in str(exc_info.value)
        assert exc_info.value.context["opset"] == invalid_opset


def test_validate_input_missing_batch_size_with_shape(
    mock_checkpoint_path: Path, mock_output_path: Path
) -> None:
    """Test validate_input raises WorkflowError when shape is provided without batch_size."""
    command = ConvertCommand()

    input_data = ConvertCommandInput(
        checkpoint_path=mock_checkpoint_path,
        output_path=mock_output_path,
        shape="3,224,224",
        batch_size=None,  # Missing batch size
        opset=17,
    )

    with pytest.raises(WorkflowError) as exc_info:
        command.validate_input(input_data, cast(Any, None))

    assert "provide --batch-size >= 1" in str(exc_info.value)
    assert exc_info.value.context["batch_size"] is None


def test_validate_input_zero_batch_size_with_shape(
    mock_checkpoint_path: Path, mock_output_path: Path
) -> None:
    """Test validate_input raises WorkflowError for zero batch size with shape."""
    command = ConvertCommand()

    input_data = ConvertCommandInput(
        checkpoint_path=mock_checkpoint_path,
        output_path=mock_output_path,
        shape="3,224,224",
        batch_size=0,
        opset=17,
    )

    with pytest.raises(WorkflowError) as exc_info:
        command.validate_input(input_data, cast(Any, None))

    assert "provide --batch-size >= 1" in str(exc_info.value)
    assert exc_info.value.context["batch_size"] == 0


def test_validate_input_negative_batch_size_with_shape(
    mock_checkpoint_path: Path, mock_output_path: Path
) -> None:
    """Test validate_input raises WorkflowError for negative batch size with shape."""
    command = ConvertCommand()

    input_data = ConvertCommandInput(
        checkpoint_path=mock_checkpoint_path,
        output_path=mock_output_path,
        shape="3,224,224",
        batch_size=-1,
        opset=17,
    )

    with pytest.raises(WorkflowError) as exc_info:
        command.validate_input(input_data, cast(Any, None))

    assert "provide --batch-size >= 1" in str(exc_info.value)
    assert exc_info.value.context["batch_size"] == -1


def test_parse_or_infer_shapes_invalid_shape_specs() -> None:
    """Test _parse_or_infer_shapes raises WorkflowError for invalid shape specs."""
    command = ConvertCommand()

    # Non-numeric and negative/zero dimension specs
    invalid_specs = [
        ("abc", "Invalid shape spec"),
        ("1,abc,3", "Invalid shape spec"),
        ("-1,28", "All shape dimensions must be positive"),
        ("0,28", "All shape dimensions must be positive"),
    ]

    for invalid_spec, expected_message_part in invalid_specs:
        with pytest.raises(WorkflowError) as exc_info:
            command._parse_or_infer_shapes(shape_spec=invalid_spec, settings=None, default_batch=1)

        assert expected_message_part in str(exc_info.value)


def test_parse_or_infer_shapes_empty_shape_specs_behavior() -> None:
    """Test _parse_or_infer_shapes behavior with empty/whitespace specs."""
    command = ConvertCommand()

    # Completely empty string triggers "no shape" path
    with pytest.raises(WorkflowError) as exc_info:
        command._parse_or_infer_shapes("", settings=None, default_batch=1)
    assert "No shape provided and no config available" in str(exc_info.value)

    # Pure whitespace returns empty shapes (edge case)
    shapes, inferred = command._parse_or_infer_shapes("  ", settings=None, default_batch=1)
    assert shapes == []
    assert not inferred

    # Comma-only specs raise "must include feature dimensions" error
    for comma_spec in [",", " , "]:
        with pytest.raises(WorkflowError) as exc_info:
            command._parse_or_infer_shapes(comma_spec, settings=None, default_batch=1)
        assert "Shape must include feature dimensions" in str(exc_info.value)

    # Semicolon-only returns empty shapes (edge case)
    shapes, inferred = command._parse_or_infer_shapes(";;", settings=None, default_batch=1)
    assert shapes == []
    assert not inferred


def test_parse_or_infer_shapes_no_shape_no_settings() -> None:
    """Test _parse_or_infer_shapes raises WorkflowError when no shape and no settings."""
    command = ConvertCommand()

    with pytest.raises(WorkflowError) as exc_info:
        command._parse_or_infer_shapes(shape_spec=None, settings=None, default_batch=1)

    assert "No shape provided and no config available" in str(exc_info.value)
    assert exc_info.value.context["command"] == "convert"


@patch("dlkit.interfaces.api.commands.convert_command.FlexibleBuildStrategy")
def test_parse_or_infer_shapes_no_dataloader_available(
    mock_strategy_class: Mock, mock_settings: Mock
) -> None:
    """Test _parse_or_infer_shapes raises WorkflowError when no dataloader is available."""
    # Setup mock to return datamodule without any working dataloaders
    mock_datamodule = Mock()
    mock_datamodule.predict_dataloader = Mock(return_value=None)
    mock_datamodule.val_dataloader = Mock(return_value=None)
    mock_datamodule.test_dataloader = Mock(return_value=None)
    mock_datamodule.train_dataloader = Mock(return_value=None)

    mock_components = Mock()
    mock_components.datamodule = mock_datamodule

    mock_strategy = Mock()
    mock_strategy.build = Mock(return_value=mock_components)
    mock_strategy_class.return_value = mock_strategy

    command = ConvertCommand()

    with pytest.raises(WorkflowError) as exc_info:
        command._parse_or_infer_shapes(shape_spec=None, settings=mock_settings, default_batch=1)

    assert "Could not construct a dataloader for shape inference" in str(exc_info.value)


@patch("dlkit.interfaces.api.commands.convert_command.FlexibleBuildStrategy")
def test_parse_or_infer_shapes_dataloader_exception(
    mock_strategy_class: Mock, mock_settings: Mock
) -> None:
    """Test _parse_or_infer_shapes raises WorkflowError when dataloader methods raise exceptions."""
    # Setup mock datamodule where all dataloader methods raise exceptions
    mock_datamodule = Mock()
    mock_datamodule.predict_dataloader = Mock(side_effect=Exception("Predict loader failed"))
    mock_datamodule.val_dataloader = Mock(side_effect=Exception("Val loader failed"))
    mock_datamodule.test_dataloader = Mock(side_effect=Exception("Test loader failed"))
    mock_datamodule.train_dataloader = Mock(side_effect=Exception("Train loader failed"))

    mock_components = Mock()
    mock_components.datamodule = mock_datamodule

    mock_strategy = Mock()
    mock_strategy.build = Mock(return_value=mock_components)
    mock_strategy_class.return_value = mock_strategy

    command = ConvertCommand()

    with pytest.raises(WorkflowError) as exc_info:
        command._parse_or_infer_shapes(shape_spec=None, settings=mock_settings, default_batch=1)

    assert "Could not construct a dataloader for shape inference" in str(exc_info.value)


@patch("dlkit.interfaces.api.commands.convert_command.FlexibleBuildStrategy")
def test_parse_or_infer_shapes_batch_iteration_fails(
    mock_strategy_class: Mock, mock_settings: Mock
) -> None:
    """Test _parse_or_infer_shapes raises WorkflowError when batch iteration fails."""
    # Setup mock dataloader that fails during iteration
    mock_dataloader = Mock()
    mock_dataloader.__iter__ = Mock(side_effect=Exception("Batch iteration failed"))

    mock_datamodule = Mock()
    mock_datamodule.predict_dataloader = Mock(return_value=mock_dataloader)

    mock_components = Mock()
    mock_components.datamodule = mock_datamodule

    mock_strategy = Mock()
    mock_strategy.build = Mock(return_value=mock_components)
    mock_strategy_class.return_value = mock_strategy

    command = ConvertCommand()

    with pytest.raises(WorkflowError) as exc_info:
        command._parse_or_infer_shapes(shape_spec=None, settings=mock_settings, default_batch=1)

    assert "Failed to get a batch from dataloader" in str(exc_info.value)


@patch("dlkit.interfaces.api.commands.convert_command.FlexibleBuildStrategy")
def test_parse_or_infer_shapes_dict_batch_no_x_key(
    mock_strategy_class: Mock, mock_settings: Mock
) -> None:
    """Test _parse_or_infer_shapes raises WorkflowError for dict batch without 'x' key."""
    # Setup dataloader that returns dict without 'x' key and no tensor values
    batch = {"y": "not_a_tensor", "z": 123}
    mock_dataloader = Mock()
    mock_dataloader.__iter__ = Mock(return_value=iter([batch]))

    mock_datamodule = Mock()
    mock_datamodule.predict_dataloader = Mock(return_value=mock_dataloader)

    mock_components = Mock()
    mock_components.datamodule = mock_datamodule

    mock_strategy = Mock()
    mock_strategy.build = Mock(return_value=mock_components)
    mock_strategy_class.return_value = mock_strategy

    command = ConvertCommand()

    with pytest.raises(WorkflowError) as exc_info:
        command._parse_or_infer_shapes(shape_spec=None, settings=mock_settings, default_batch=1)

    assert "Could not find input tensor 'x' in batch" in str(exc_info.value)


@patch("dlkit.interfaces.api.commands.convert_command.FlexibleBuildStrategy")
def test_parse_or_infer_shapes_tuple_batch_no_shape(
    mock_strategy_class: Mock, mock_settings: Mock
) -> None:
    """Test _parse_or_infer_shapes raises WorkflowError for tuple batch without shape."""
    # Setup dataloader that returns tuple with object that has no shape
    batch = ("not_a_tensor", "target")
    mock_dataloader = Mock()
    mock_dataloader.__iter__ = Mock(return_value=iter([batch]))

    mock_datamodule = Mock()
    mock_datamodule.predict_dataloader = Mock(return_value=mock_dataloader)

    mock_components = Mock()
    mock_components.datamodule = mock_datamodule

    mock_strategy = Mock()
    mock_strategy.build = Mock(return_value=mock_components)
    mock_strategy_class.return_value = mock_strategy

    command = ConvertCommand()

    with pytest.raises(WorkflowError) as exc_info:
        command._parse_or_infer_shapes(shape_spec=None, settings=mock_settings, default_batch=1)

    assert "First element of batch has no shape" in str(exc_info.value)


@patch("dlkit.interfaces.api.commands.convert_command.FlexibleBuildStrategy")
def test_parse_or_infer_shapes_single_batch_no_shape(
    mock_strategy_class: Mock, mock_settings: Mock
) -> None:
    """Test _parse_or_infer_shapes raises WorkflowError for single batch without shape."""
    # Setup dataloader that returns object without shape attribute
    batch = "not_a_tensor"
    mock_dataloader = Mock()
    mock_dataloader.__iter__ = Mock(return_value=iter([batch]))

    mock_datamodule = Mock()
    mock_datamodule.predict_dataloader = Mock(return_value=mock_dataloader)

    mock_components = Mock()
    mock_components.datamodule = mock_datamodule

    mock_strategy = Mock()
    mock_strategy.build = Mock(return_value=mock_components)
    mock_strategy_class.return_value = mock_strategy

    command = ConvertCommand()

    with pytest.raises(WorkflowError) as exc_info:
        command._parse_or_infer_shapes(shape_spec=None, settings=mock_settings, default_batch=1)

    assert "Batch object is not a Tensor and not a supported container" in str(exc_info.value)


@patch(
    "dlkit.interfaces.api.commands.convert_command.WrapperFactory.create_wrapper_from_checkpoint"
)
def test_execute_no_input_shapes(
    mock_wrapper_factory: Mock,
    mock_checkpoint_path: Path,
    mock_output_path: Path,
    mock_wrapper: Mock,
    mock_settings: Mock,
) -> None:
    """Test execute raises WorkflowError when no input shapes can be determined."""
    mock_wrapper_factory.return_value = mock_wrapper

    # Mock _parse_or_infer_shapes to return empty shapes
    command = ConvertCommand()
    with patch.object(command, "_parse_or_infer_shapes", return_value=([], False)):
        input_data = ConvertCommandInput(
            checkpoint_path=mock_checkpoint_path,
            output_path=mock_output_path,
            shape=None,
            batch_size=None,
            opset=17,
        )

        with pytest.raises(WorkflowError) as exc_info:
            command.execute(input_data, mock_settings)

        assert "Could not determine input shape" in str(exc_info.value)


@patch(
    "dlkit.interfaces.api.commands.convert_command.WrapperFactory.create_wrapper_from_checkpoint"
)
def test_execute_batch_size_mismatch(
    mock_wrapper_factory: Mock,
    mock_checkpoint_path: Path,
    mock_output_path: Path,
    mock_wrapper: Mock,
    mock_settings: Mock,
) -> None:
    """Test execute raises WorkflowError for batch size mismatch."""
    mock_wrapper_factory.return_value = mock_wrapper

    # Mock _parse_or_infer_shapes to return shapes with different batch size
    inferred_shapes = [(8, 3, 224, 224)]  # Batch size 8
    command = ConvertCommand()
    with patch.object(command, "_parse_or_infer_shapes", return_value=(inferred_shapes, True)):
        input_data = ConvertCommandInput(
            checkpoint_path=mock_checkpoint_path,
            output_path=mock_output_path,
            shape=None,
            batch_size=4,  # Different batch size
            opset=17,
        )

        with pytest.raises(WorkflowError) as exc_info:
            command.execute(input_data, mock_settings)

        assert "Batch size mismatch" in str(exc_info.value)
        assert exc_info.value.context["expected_batch"] == 4
        assert exc_info.value.context["found_batches"] == [8]


@patch(
    "dlkit.interfaces.api.commands.convert_command.WrapperFactory.create_wrapper_from_checkpoint"
)
def test_execute_inconsistent_batch_sizes_in_shapes(
    mock_wrapper_factory: Mock,
    mock_checkpoint_path: Path,
    mock_output_path: Path,
    mock_wrapper: Mock,
    mock_settings: Mock,
) -> None:
    """Test execute raises WorkflowError for inconsistent batch sizes in multiple shapes."""
    mock_wrapper_factory.return_value = mock_wrapper

    # Mock _parse_or_infer_shapes to return shapes with inconsistent batch sizes
    inferred_shapes = [(4, 3, 224, 224), (8, 100)]  # Different batch sizes
    command = ConvertCommand()
    with patch.object(command, "_parse_or_infer_shapes", return_value=(inferred_shapes, True)):
        input_data = ConvertCommandInput(
            checkpoint_path=mock_checkpoint_path,
            output_path=mock_output_path,
            shape=None,
            batch_size=4,
            opset=17,
        )

        with pytest.raises(WorkflowError) as exc_info:
            command.execute(input_data, mock_settings)

        assert "Batch size mismatch" in str(exc_info.value)
        assert exc_info.value.context["expected_batch"] == 4
        assert sorted(exc_info.value.context["found_batches"]) == [4, 8]


@patch(
    "dlkit.interfaces.api.commands.convert_command.WrapperFactory.create_wrapper_from_checkpoint"
)
def test_execute_wrapper_factory_exception(
    mock_wrapper_factory: Mock, valid_convert_input: ConvertCommandInput, mock_settings: Mock
) -> None:
    """Test execute converts WrapperFactory exceptions to WorkflowError."""
    # Mock wrapper factory to raise an exception
    mock_wrapper_factory.side_effect = Exception("Failed to load checkpoint")

    command = ConvertCommand()

    with pytest.raises(WorkflowError) as exc_info:
        command.execute(valid_convert_input, mock_settings)

    assert "Conversion failed" in str(exc_info.value)
    assert exc_info.value.context["command"] == "convert"
    assert exc_info.value.context["error_type"] == "Exception"


@patch("dlkit.interfaces.api.commands.convert_command.torch.onnx.export")
@patch("dlkit.interfaces.api.commands.convert_command.torch.ones")
@patch(
    "dlkit.interfaces.api.commands.convert_command.WrapperFactory.create_wrapper_from_checkpoint"
)
def test_execute_torch_export_exception(
    mock_wrapper_factory: Mock,
    mock_torch_ones: Mock,
    mock_torch_export: Mock,
    valid_convert_input: ConvertCommandInput,
    mock_wrapper: Mock,
    mock_settings: Mock,
) -> None:
    """Test execute converts torch.onnx.export exceptions to WorkflowError."""
    # Setup mocks
    mock_wrapper_factory.return_value = mock_wrapper
    mock_torch_ones.return_value = Mock()
    mock_torch_export.side_effect = Exception("ONNX export failed")

    command = ConvertCommand()

    with pytest.raises(WorkflowError) as exc_info:
        command.execute(valid_convert_input, mock_settings)

    assert "Conversion failed" in str(exc_info.value)
    assert exc_info.value.context["command"] == "convert"
    assert exc_info.value.context["error_type"] == "Exception"


def test_execute_preserves_workflow_errors(
    mock_checkpoint_path: Path, tmp_path: Path, mock_settings: Mock
) -> None:
    """Test execute preserves WorkflowError exceptions without wrapping."""
    # Create input that will fail validation (directory as output)
    output_dir = tmp_path / "output_dir"
    output_dir.mkdir()

    input_data = ConvertCommandInput(
        checkpoint_path=mock_checkpoint_path,
        output_path=output_dir,
        shape="3,224,224",
        batch_size=4,
        opset=17,
    )

    command = ConvertCommand()

    with pytest.raises(WorkflowError) as exc_info:
        command.execute(input_data, mock_settings)

    # Should be the original WorkflowError from validation, not wrapped
    assert "Output path points to a directory" in str(exc_info.value)
    assert exc_info.value.context["command"] == "convert"
