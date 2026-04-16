"""Fixtures for neural network tests.

All fixtures use minimal/lightweight dimensions:
- Batch size: 3
- Features: 2-4
- Sequence length: 8-16
- Latent: 2-4
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def batch_size() -> int:
    """Standard batch size for tests."""
    return 3


@pytest.fixture
def dense_input(batch_size: int) -> torch.Tensor:
    """2-feature input for dense/linear models.

    Shape: (batch_size, 2)
    """
    return torch.randn(batch_size, 2)


@pytest.fixture
def conv_input(batch_size: int) -> torch.Tensor:
    """2-channel, 16-timestep input for 1D conv models.

    Shape: (batch_size, 2, 16)
    """
    return torch.randn(batch_size, 2, 16)


@pytest.fixture
def latent_input(batch_size: int) -> torch.Tensor:
    """4-dim latent vector for decoder/bottleneck modules.

    Shape: (batch_size, 4)
    """
    return torch.randn(batch_size, 4)


@pytest.fixture
def temporal_input(batch_size: int) -> torch.Tensor:
    """2-channel, 8-timestep temporal input for attention blocks.

    Shape: (batch_size, 2, 8)
    """
    return torch.randn(batch_size, 2, 8)
