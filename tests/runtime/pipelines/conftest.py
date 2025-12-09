"""Shared fixtures for processing module tests.

This module provides modular, composable fixtures for testing the
processing pipeline components following SOLID principles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dlkit.runtime.pipelines.interfaces import DataProvider
from dlkit.runtime.pipelines.providers import FileDataProvider
from dlkit.tools.config.data_entries import (
    DataEntry,
    Feature,
    Target,
    Latent,
    Prediction,
    PathFeature,
    PathTarget,
    ValueFeature,
    ValueTarget,
    is_feature_entry,
    is_target_entry,
)


# Test dataflow constants
SAMPLE_DATA_SIZE = (10, 3)  # 10 samples, 3 features each
TEST_DTYPE = torch.float32
TEST_INDICES = [0, 5, 9]  # Valid indices for testing


@pytest.fixture
def sample_numpy_data() -> np.ndarray:
    """Generate sample numpy dataflow for testing.

    Returns:
        np.ndarray: Sample dataflow with known values for testing
    """
    np.random.seed(42)  # Fixed seed for reproducible tests
    return np.random.randn(*SAMPLE_DATA_SIZE).astype(np.float32)


@pytest.fixture
def sample_torch_data() -> torch.Tensor:
    """Generate sample torch tensor dataflow for testing.

    Returns:
        torch.Tensor: Sample tensor dataflow with known values
    """
    torch.manual_seed(42)  # Fixed seed for reproducible tests
    return torch.randn(*SAMPLE_DATA_SIZE, dtype=TEST_DTYPE)


@pytest.fixture
def test_npy_file(tmp_path: Path, sample_numpy_data: np.ndarray) -> Path:
    """Create a temporary .npy file with sample

    Args:
        tmp_path: Pytest temporary directory fixture
        sample_numpy_data: Sample numpy dataflow to save

    Returns:
        Path: Path to the created .npy file
    """
    file_path = tmp_path / "test_data.npy"
    np.save(file_path, sample_numpy_data)
    return file_path


@pytest.fixture
def test_pt_file(tmp_path: Path, sample_torch_data: torch.Tensor) -> Path:
    """Create a temporary .pt file with sample tensor

    Args:
        tmp_path: Pytest temporary directory fixture
        sample_torch_data: Sample torch tensor to save

    Returns:
        Path: Path to the created .pt file
    """
    file_path = tmp_path / "test_data.pt"
    torch.save(sample_torch_data, file_path)
    return file_path


@pytest.fixture
def test_txt_file(tmp_path: Path, sample_numpy_data: np.ndarray) -> Path:
    """Create a temporary .txt file with sample

    Args:
        tmp_path: Pytest temporary directory fixture
        sample_numpy_data: Sample numpy dataflow to save as text

    Returns:
        Path: Path to the created .txt file
    """
    file_path = tmp_path / "test_data.txt"
    np.savetxt(file_path, sample_numpy_data)
    return file_path


@pytest.fixture
def feature_entry(test_npy_file: Path) -> Feature:
    """Create a Feature dataflow entry with test file.

    Args:
        test_npy_file: Path to test .npy file

    Returns:
        Feature: Configured feature entry for testing
    """
    return Feature(name="test_feature", path=test_npy_file, dtype=TEST_DTYPE, transforms=[])


@pytest.fixture
def target_entry(test_pt_file: Path) -> Target:
    """Create a Target dataflow entry with test file.

    Args:
        test_pt_file: Path to test .pt file

    Returns:
        Target: Configured target entry for testing
    """
    return Target(
        name="test_target", path=test_pt_file, dtype=TEST_DTYPE, write=True, transforms=[]
    )


@pytest.fixture
def latent_entry() -> Latent:
    """Create a Latent dataflow entry (no file path).

    Returns:
        Latent: Configured latent entry for testing
    """
    return Latent(name="test_latent", dtype=TEST_DTYPE, write=False)


@pytest.fixture
def prediction_entry() -> Prediction:
    """Create a Prediction dataflow entry.

    Returns:
        Prediction: Configured prediction entry for testing
    """
    return Prediction(
        name="test_prediction", target_name="test_target", dtype=TEST_DTYPE, write=True
    )


@pytest.fixture
def file_data_provider() -> FileDataProvider:
    """Create a FileDataProvider instance for testing.

    Returns:
        FileDataProvider: Fresh provider instance with empty caches
    """
    return FileDataProvider()


@pytest.fixture
def populated_provider(
    file_data_provider: FileDataProvider, feature_entry: Feature, target_entry: Target
) -> FileDataProvider:
    """Create a FileDataProvider pre-populated with test

    Args:
        file_data_provider: Fresh provider instance
        feature_entry: Feature entry to load
        target_entry: Target entry to load

    Returns:
        FileDataProvider: Provider with pre-loaded test dataflow
    """
    # Pre-load dataflow to populate caches
    file_data_provider.get_length(feature_entry)
    file_data_provider.get_length(target_entry)
    return file_data_provider


@pytest.fixture
def mock_provider() -> DataProvider:
    """Create a mock DataProvider for registry testing.

    Returns:
        DataProvider: Mock provider implementation
    """

    class MockProvider(DataProvider):
        def __init__(self, handles_features: bool = True, handles_targets: bool = False):
            self.handles_features = handles_features
            self.handles_targets = handles_targets

        def can_handle(self, entry: DataEntry) -> bool:
            if is_feature_entry(entry):
                return self.handles_features
            elif is_target_entry(entry):
                return self.handles_targets
            return False

        def load_data(self, entry: DataEntry, idx: int) -> torch.Tensor:
            return torch.zeros(3, dtype=TEST_DTYPE)

        def get_length(self, entry: DataEntry) -> int:
            return 10

    return MockProvider()


@pytest.fixture
def multi_provider_setup() -> list[DataProvider]:
    """Create multiple providers for registry testing.

    Returns:
        list[DataProvider]: List of mock providers with different capabilities
    """

    class FeatureProvider(DataProvider):
        def can_handle(self, entry: DataEntry) -> bool:
            return is_feature_entry(entry)

        def load_data(self, entry: DataEntry, idx: int) -> torch.Tensor:
            return torch.ones(3, dtype=TEST_DTYPE)

        def get_length(self, entry: DataEntry) -> int:
            return 5

    class TargetProvider(DataProvider):
        def can_handle(self, entry: DataEntry) -> bool:
            return is_target_entry(entry)

        def load_data(self, entry: DataEntry, idx: int) -> torch.Tensor:
            return torch.zeros(3, dtype=TEST_DTYPE)

        def get_length(self, entry: DataEntry) -> int:
            return 5

    return [FeatureProvider(), TargetProvider()]


@pytest.fixture
def nonexistent_file(tmp_path: Path) -> Path:
    """Create a path to a non-existent file for error testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path: Path to a file that doesn't exist
    """
    return tmp_path / "nonexistent.npy"


@pytest.fixture
def invalid_feature_entry(tmp_path: Path) -> Feature:
    """Create a Feature entry pointing to a file that will be deleted.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Feature: Feature entry that will cause loading errors
    """
    # Create a temporary file first to satisfy Pydantic validation
    temp_file = tmp_path / "temp_file.npy"
    temp_file.write_text("dummy")

    # Create the feature entry
    feature = Feature(name="invalid_feature", path=temp_file, dtype=TEST_DTYPE, transforms=[])

    # Now delete the file to simulate a missing file scenario
    temp_file.unlink()

    return feature
