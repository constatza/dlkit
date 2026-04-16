"""Fixtures for spectral / frequency-domain layer tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def batch_size() -> int:
    """Standard batch size."""
    return 4


@pytest.fixture
def n_channels() -> int:
    """Channel count used in conv-style tests."""
    return 8


@pytest.fixture
def spatial_length() -> int:
    """Spatial sequence length."""
    return 32


@pytest.fixture
def n_features() -> int:
    """Flat feature count for FFNN-style tests."""
    return 16


@pytest.fixture
def n_modes() -> int:
    """Default number of Fourier modes."""
    return 6


@pytest.fixture
def conv_input(batch_size: int, n_channels: int, spatial_length: int) -> torch.Tensor:
    """Conv-style input: (batch, channels, length).

    Shape: (4, 8, 32)
    """
    return torch.randn(batch_size, n_channels, spatial_length)


@pytest.fixture
def flat_input(batch_size: int, n_features: int) -> torch.Tensor:
    """Flat feature input: (batch, features).

    Shape: (4, 16)
    """
    return torch.randn(batch_size, n_features)
