"""Fixtures for callbacks testing."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer


# Constants for test configuration
DEFAULT_BATCH_SIZE = 4
DEFAULT_FEATURE_SIZE = 10
DEFAULT_NUM_CLASSES = 3


@pytest.fixture
def mock_mlflow_adapter() -> Mock:
    """Create a mock MLflow adapter for testing.

    Returns:
        Mock: MLflow adapter mock with necessary methods
    """
    adapter = Mock()
    adapter.get_artifact_uri.return_value = None
    adapter.get_active_run.return_value = None
    adapter.log_artifact.return_value = None
    return adapter


def mock_mlflow_adapter_with_uri_factory(tmp_path: Path) -> Mock:
    """Create a mock MLflow adapter that returns an artifact URI.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        Mock: MLflow adapter mock configured with artifact URI
    """
    adapter = Mock()
    # Use file:// protocol for the artifact URI
    artifact_uri = f"file://{tmp_path / 'mlflow_artifacts'}"
    adapter.get_artifact_uri.return_value = artifact_uri

    # Mock active run for artifact logging
    mock_run = Mock()
    mock_run.info.run_id = "test_run_123"
    adapter.get_active_run.return_value = mock_run
    adapter.log_artifact.return_value = None

    return adapter


@pytest.fixture
def sample_tensor_dict() -> dict[str, torch.Tensor]:
    """Create sample tensor dictionary for testing.

    Returns:
        dict: Dictionary with string keys and tensor values
    """
    return {
        "predictions": torch.randn(DEFAULT_BATCH_SIZE, DEFAULT_NUM_CLASSES),
        "features": torch.randn(DEFAULT_BATCH_SIZE, DEFAULT_FEATURE_SIZE),
        "logits": torch.randn(DEFAULT_BATCH_SIZE, DEFAULT_NUM_CLASSES),
    }


@pytest.fixture
def sample_tensor_list() -> list[torch.Tensor]:
    """Create sample tensor list for testing.

    Returns:
        list: List of tensors with different shapes
    """
    return [
        torch.randn(DEFAULT_BATCH_SIZE, DEFAULT_NUM_CLASSES),
        torch.randn(DEFAULT_BATCH_SIZE, DEFAULT_FEATURE_SIZE),
        torch.randn(DEFAULT_BATCH_SIZE, 1),
    ]


@pytest.fixture
def sample_single_tensor() -> torch.Tensor:
    """Create single tensor for testing.

    Returns:
        torch.Tensor: Single tensor for testing
    """
    return torch.randn(DEFAULT_BATCH_SIZE, DEFAULT_NUM_CLASSES)


@pytest.fixture
def batch_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample batch dataflow for testing.

    Returns:
        tuple: Sample (input, target) batch dataflow
    """
    inputs = torch.randn(DEFAULT_BATCH_SIZE, DEFAULT_FEATURE_SIZE)
    targets = torch.randint(0, DEFAULT_NUM_CLASSES, (DEFAULT_BATCH_SIZE,))
    return inputs, targets


@pytest.fixture
def mock_trainer() -> Mock:
    """Create a mock PyTorch Lightning Trainer.

    Returns:
        Mock: Trainer mock with proper method signatures
    """
    trainer = Mock(spec=Trainer)
    trainer.current_epoch = 0
    trainer.global_step = 0
    return trainer


@pytest.fixture
def mock_lightning_module() -> Mock:
    """Create a mock PyTorch Lightning module.

    Returns:
        Mock: LightningModule mock with proper method signatures
    """
    module = Mock(spec=LightningModule)
    module.device = torch.device("cpu")
    return module


@pytest.fixture
def custom_filenames() -> tuple[str, ...]:
    """Custom filenames for testing filename override functionality.

    Returns:
        tuple: Custom filename sequence
    """
    return ("custom_pred", "custom_feat", "custom_logits")


@pytest.fixture
def multi_batch_tensor_data() -> list[dict[str, torch.Tensor]]:
    """Create multiple batches of tensor dataflow for accumulation testing.

    Returns:
        list: Multiple batches of tensor dictionaries
    """
    batch_size = 2  # Smaller batches for testing
    return [
        {
            "predictions": torch.randn(batch_size, DEFAULT_NUM_CLASSES),
            "features": torch.randn(batch_size, DEFAULT_FEATURE_SIZE),
        }
        for _ in range(3)  # 3 batches
    ]


@pytest.fixture
def expected_concatenated_shapes() -> dict[str, tuple[int, ...]]:
    """Expected shapes after concatenating multi-batch

    Returns:
        dict: Expected tensor shapes after concatenation
    """
    return {
        "predictions": (6, DEFAULT_NUM_CLASSES),  # 3 batches * 2 samples each
        "features": (6, DEFAULT_FEATURE_SIZE),
    }


@pytest.fixture
def invalid_outputs() -> list[Any]:
    """Invalid output types for error testing.

    Returns:
        list: Various invalid output types
    """
    return [
        "string_output",
        123,
        {"key": "not_a_tensor"},
        None,
        object(),
    ]


@pytest.fixture
def mixed_valid_invalid_dict() -> dict[str, Any]:
    """Dictionary with mix of valid tensors and invalid values.

    Returns:
        dict: Mixed valid/invalid output dictionary
    """
    return {
        "valid_tensor": torch.randn(DEFAULT_BATCH_SIZE, DEFAULT_NUM_CLASSES),
        "invalid_string": "not_a_tensor",
        "invalid_int": 42,
        "valid_tensor2": torch.randn(DEFAULT_BATCH_SIZE, DEFAULT_FEATURE_SIZE),
    }


def output_dir_permissions_error_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create directory that will cause permission errors during save.

    Args:
        tmp_path: Pytest temporary path fixture
        monkeypatch: Pytest monkeypatch fixture

    Returns:
        Path: Directory path that will cause permission errors
    """
    error_dir = tmp_path / "permission_error"
    error_dir.mkdir()

    # Mock numpy.save to raise OSError
    def mock_save(*args, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr("numpy.save", mock_save)
    return error_dir
