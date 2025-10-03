"""Fixtures for metrics tests."""

import pytest
import torch
from torch import Tensor


@pytest.fixture
def sample_predictions() -> Tensor:
    """Sample prediction tensor for testing."""
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


@pytest.fixture
def sample_targets() -> Tensor:
    """Sample target tensor for testing."""
    return torch.tensor([[1.1, 1.9, 3.1], [3.9, 5.1, 5.9], [7.1, 7.9, 9.1]])


@pytest.fixture
def sample_2d_vectors() -> tuple[Tensor, Tensor]:
    """Sample 2D vector dataflow for normalized vector norm error testing."""
    predictions = torch.tensor([
        [1.0, 0.0],  # Predicted vector 1
        [0.0, 2.0],  # Predicted vector 2
        [1.0, 1.0],  # Predicted vector 3
    ])

    targets = torch.tensor([
        [1.0, 1.0],  # Target vector 1 (norm = sqrt(2))
        [2.0, 0.0],  # Target vector 2 (norm = 2.0)
        [0.0, 1.0],  # Target vector 3 (norm = 1.0)
    ])

    return predictions, targets


@pytest.fixture
def temporal_data() -> tuple[Tensor, Tensor]:
    """Sample temporal dataflow for time series testing."""
    # Shape: (batch, features, time)
    predictions = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0]],  # Batch 1
        [[2.0, 3.0, 4.0, 5.0]],  # Batch 2
    ])

    targets = torch.tensor([
        [[1.1, 1.9, 3.1, 3.9]],  # Batch 1
        [[2.1, 2.9, 4.1, 4.9]],  # Batch 2
    ])

    return predictions, targets


@pytest.fixture
def zero_targets() -> Tensor:
    """Zero target tensor for numerical stability testing."""
    return torch.zeros(3, 3)


@pytest.fixture
def unit_vectors() -> tuple[Tensor, Tensor]:
    """Unit vector dataflow for normalized testing."""
    predictions = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.707, 0.707],  # Approximate unit vector
    ])

    targets = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    return predictions, targets
