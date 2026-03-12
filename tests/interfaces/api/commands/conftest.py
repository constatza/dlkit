"""Fixtures for API command tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from dlkit.interfaces.api.commands.convert_command import ConvertCommandInput, ConvertResult
from dlkit.tools.config import GeneralSettings
from dlkit.core.models.wrappers.base import ProcessingLightningWrapper


# Test dataflow constants
VALID_OPSET_VERSIONS = [9, 11, 13, 17, 18]
INVALID_OPSET_VERSIONS = [1, 5, 8]
VALID_BATCH_SIZES = [1, 4, 8, 16, 32]
SAMPLE_SHAPE_SPECS = [
    "28,28",  # 2D image features
    "3,224,224",  # RGB image
    "100",  # 1D features
    "10,20,30",  # 3D features
]
INVALID_SHAPE_SPECS = [
    "",  # empty
    "abc",  # non-numeric
    "-1,28",  # negative dimension
    "0,28",  # zero dimension
]


@pytest.fixture
def mock_checkpoint_path(tmp_path: Path) -> Path:
    """Create a mock checkpoint file.

    Args:
        tmp_path: Pytest temp directory fixture

    Returns:
        Path: Path to mock checkpoint file
    """
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("mock checkpoint dataflow")
    return checkpoint


@pytest.fixture
def mock_output_path(tmp_path: Path) -> Path:
    """Create a mock output path.

    Args:
        tmp_path: Pytest temp directory fixture

    Returns:
        Path: Path for output file
    """
    return tmp_path / "model.onnx"


@pytest.fixture
def valid_convert_input(mock_checkpoint_path: Path, mock_output_path: Path) -> ConvertCommandInput:
    """Create valid ConvertCommandInput.

    Args:
        mock_checkpoint_path: Mock checkpoint file
        mock_output_path: Mock output path

    Returns:
        ConvertCommandInput: Valid input dataflow for testing
    """
    return ConvertCommandInput(
        checkpoint_path=mock_checkpoint_path,
        output_path=mock_output_path,
        shape="3,224,224",
        batch_size=4,
        opset=17,
    )


@pytest.fixture
def valid_convert_input_no_shape(
    mock_checkpoint_path: Path, mock_output_path: Path
) -> ConvertCommandInput:
    """Create valid ConvertCommandInput without shape specification.

    Args:
        mock_checkpoint_path: Mock checkpoint file
        mock_output_path: Mock output path

    Returns:
        ConvertCommandInput: Valid input dataflow without shape
    """
    return ConvertCommandInput(
        checkpoint_path=mock_checkpoint_path,
        output_path=mock_output_path,
        shape=None,
        batch_size=None,
        opset=17,
    )


@pytest.fixture
def expected_convert_result(mock_output_path: Path) -> ConvertResult:
    """Create expected ConvertResult.

    Args:
        mock_output_path: Mock output path

    Returns:
        ConvertResult: Expected result dataflow
    """
    return ConvertResult(output_path=mock_output_path, opset=17, inputs=[(4, 3, 224, 224)])


@pytest.fixture
def mock_wrapper() -> Mock:
    """Create mock Lightning wrapper.

    Returns:
        Mock: Mock wrapper with required methods
    """
    wrapper = Mock(spec=ProcessingLightningWrapper)
    wrapper.eval = Mock()
    return wrapper


@pytest.fixture
def mock_torch_tensor() -> Mock:
    """Create mock torch tensor.

    Returns:
        Mock: Mock tensor with shape attribute
    """
    tensor = Mock(spec=torch.Tensor)
    tensor.shape = (4, 3, 224, 224)
    return tensor


@pytest.fixture
def mock_dataloader_batch_dict(mock_torch_tensor: Mock) -> dict[str, Any]:
    """Create mock dataloader batch as dictionary.

    Args:
        mock_torch_tensor: Mock tensor

    Returns:
        dict: Mock batch dataflow as dict
    """
    return {"x": mock_torch_tensor, "y": Mock()}


@pytest.fixture
def mock_dataloader_batch_tuple(mock_torch_tensor: Mock) -> tuple[Any, Any]:
    """Create mock dataloader batch as tuple.

    Args:
        mock_torch_tensor: Mock tensor

    Returns:
        tuple: Mock batch dataflow as tuple
    """
    return (mock_torch_tensor, Mock())


@pytest.fixture
def mock_dataloader_batch_tensor(mock_torch_tensor: Mock) -> Mock:
    """Create mock dataloader batch as single tensor.

    Args:
        mock_torch_tensor: Mock tensor

    Returns:
        Mock: Mock batch dataflow as single tensor
    """
    return mock_torch_tensor


@pytest.fixture
def mock_dataloader(mock_dataloader_batch_dict: dict[str, Any]) -> Mock:
    """Create mock dataloader.

    Args:
        mock_dataloader_batch_dict: Mock batch dataflow

    Returns:
        Mock: Mock dataloader
    """
    dataloader = Mock()
    dataloader.__iter__ = Mock(return_value=iter([mock_dataloader_batch_dict]))
    return dataloader


@pytest.fixture
def mock_datamodule(mock_dataloader: Mock) -> Mock:
    """Create mock datamodule with dataloader methods.

    Args:
        mock_dataloader: Mock dataloader

    Returns:
        Mock: Mock datamodule
    """
    datamodule = Mock()
    datamodule.predict_dataloader = Mock(return_value=mock_dataloader)
    datamodule.val_dataloader = Mock(return_value=mock_dataloader)
    datamodule.test_dataloader = Mock(return_value=mock_dataloader)
    datamodule.train_dataloader = Mock(return_value=mock_dataloader)
    return datamodule


@pytest.fixture
def mock_build_components(mock_datamodule: Mock) -> Mock:
    """Create mock build components.

    Args:
        mock_datamodule: Mock datamodule

    Returns:
        Mock: Mock components with datamodule
    """
    components = Mock()
    components.datamodule = mock_datamodule
    return components


@pytest.fixture
def mock_build_strategy(mock_build_components: Mock) -> Mock:
    """Create mock build strategy.

    Args:
        mock_build_components: Mock components

    Returns:
        Mock: Mock strategy
    """
    strategy = Mock()
    strategy.build = Mock(return_value=mock_build_components)
    return strategy


@pytest.fixture
def mock_settings() -> Mock:
    """Create mock GeneralSettings.

    Returns:
        Mock: Mock settings
    """
    return Mock(spec=GeneralSettings)


# Validation command fixtures


@pytest.fixture
def valid_training_settings() -> Mock:
    """Create mock settings for valid training configuration.

    Returns:
        Mock: Mock settings with all required sections for training
    """
    settings = Mock(spec=GeneralSettings)
    settings.MODEL = Mock()
    settings.DATASET = Mock()
    settings.DATAMODULE = Mock()
    settings.TRAINING = Mock()
    settings.SESSION = Mock()
    settings.SESSION.inference = False
    settings.MLFLOW = None
    settings.OPTUNA = Mock()
    settings.OPTUNA.enabled = False
    return settings


@pytest.fixture
def valid_inference_settings() -> Mock:
    """Create mock settings for valid inference configuration.

    Returns:
        Mock: Mock settings with all required sections for inference
    """
    settings = Mock(spec=GeneralSettings)
    settings.MODEL = Mock()
    settings.MODEL.checkpoint = "/path/to/checkpoint.ckpt"
    settings.DATASET = Mock()
    settings.DATAMODULE = Mock()
    settings.TRAINING = None
    settings.SESSION = Mock()
    settings.SESSION.inference = True
    settings.MLFLOW = None
    settings.OPTUNA = Mock()
    settings.OPTUNA.enabled = False
    return settings


@pytest.fixture
def mlflow_active_settings() -> Mock:
    """Create mock settings with MLflow active.

    Returns:
        Mock: Mock settings with MLflow enabled
    """
    settings = Mock(spec=GeneralSettings)
    settings.MODEL = Mock()
    settings.DATASET = Mock()
    settings.DATAMODULE = Mock()
    settings.TRAINING = Mock()
    settings.SESSION = Mock()
    settings.SESSION.inference = False
    settings.MLFLOW = Mock()
    settings.OPTUNA = Mock()
    settings.OPTUNA.enabled = False
    return settings


@pytest.fixture
def optuna_active_settings() -> Mock:
    """Create mock settings with Optuna active.

    Returns:
        Mock: Mock settings with Optuna enabled
    """
    settings = Mock(spec=GeneralSettings)
    settings.MODEL = Mock()
    settings.DATASET = Mock()
    settings.DATAMODULE = Mock()
    settings.TRAINING = Mock()
    settings.SESSION = Mock()
    settings.SESSION.inference = False
    settings.MLFLOW = None
    settings.OPTUNA = Mock()
    settings.OPTUNA.enabled = True
    return settings


@pytest.fixture
def missing_model_settings() -> Mock:
    """Create mock settings missing MODEL section.

    Returns:
        Mock: Mock settings without MODEL section
    """
    settings = Mock(spec=GeneralSettings)
    settings.MODEL = None
    settings.DATASET = Mock()
    settings.DATAMODULE = Mock()
    settings.TRAINING = Mock()
    settings.SESSION = Mock()
    settings.SESSION.inference = False
    settings.MLFLOW = None
    settings.OPTUNA = Mock()
    settings.OPTUNA.enabled = False
    return settings


@pytest.fixture
def missing_dataset_settings() -> Mock:
    """Create mock settings missing DATASET section.

    Returns:
        Mock: Mock settings without DATASET section
    """
    settings = Mock(spec=GeneralSettings)
    settings.MODEL = Mock()
    settings.DATASET = None
    settings.DATAMODULE = Mock()
    settings.TRAINING = Mock()
    settings.SESSION = Mock()
    settings.SESSION.inference = False
    settings.MLFLOW = None
    settings.OPTUNA = Mock()
    settings.OPTUNA.enabled = False
    return settings


@pytest.fixture
def missing_datamodule_settings() -> Mock:
    """Create mock settings missing DATAMODULE section.

    Returns:
        Mock: Mock settings without DATAMODULE section
    """
    settings = Mock(spec=GeneralSettings)
    settings.MODEL = Mock()
    settings.DATASET = Mock()
    settings.DATAMODULE = None
    settings.TRAINING = Mock()
    settings.SESSION = Mock()
    settings.SESSION.inference = False
    settings.MLFLOW = None
    settings.OPTUNA = Mock()
    settings.OPTUNA.enabled = False
    return settings


@pytest.fixture
def missing_training_settings() -> Mock:
    """Create mock settings missing TRAINING section for training mode.

    Returns:
        Mock: Mock settings without TRAINING section in training mode
    """
    settings = Mock(spec=GeneralSettings)
    settings.MODEL = Mock()
    settings.DATASET = Mock()
    settings.DATAMODULE = Mock()
    settings.TRAINING = None
    settings.SESSION = Mock()
    settings.SESSION.inference = False
    settings.MLFLOW = None
    settings.OPTUNA = Mock()
    settings.OPTUNA.enabled = False
    return settings


@pytest.fixture
def missing_checkpoint_inference_settings() -> Mock:
    """Create mock settings missing checkpoint for inference mode.

    Returns:
        Mock: Mock settings without MODEL.checkpoint in inference mode
    """
    settings = Mock(spec=GeneralSettings)
    settings.MODEL = Mock()
    settings.MODEL.checkpoint = None
    settings.DATASET = Mock()
    settings.DATAMODULE = Mock()
    settings.TRAINING = None
    settings.SESSION = Mock()
    settings.SESSION.inference = True
    settings.MLFLOW = None
    settings.OPTUNA = Mock()
    settings.OPTUNA.enabled = False
    return settings
