"""Fixtures for neural operator tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def batch_size() -> int:
    """Standard batch size."""
    return 4


@pytest.fixture
def n_channels() -> int:
    """Input/output channel count for grid operators."""
    return 2


@pytest.fixture
def spatial_length() -> int:
    """Grid resolution for FNO tests."""
    return 32


@pytest.fixture
def n_sensors() -> int:
    """Number of sensor locations for DeepONet branch input."""
    return 20


@pytest.fixture
def n_queries() -> int:
    """Number of query points for DeepONet trunk input."""
    return 10


@pytest.fixture
def n_coords() -> int:
    """Spatial coordinate dimension for query points."""
    return 1


@pytest.fixture
def fno_input(batch_size: int, n_channels: int, spatial_length: int) -> torch.Tensor:
    """Grid input for FNO: (batch, channels, length).

    Shape: (4, 2, 32)
    """
    return torch.randn(batch_size, n_channels, spatial_length)


@pytest.fixture
def deeponet_branch(batch_size: int, n_sensors: int) -> torch.Tensor:
    """Branch input for DeepONet: (batch, n_sensors).

    Shape: (4, 20)
    """
    return torch.randn(batch_size, n_sensors)


@pytest.fixture
def deeponet_trunk(batch_size: int, n_queries: int, n_coords: int) -> torch.Tensor:
    """Trunk query points for DeepONet: (batch, n_queries, n_coords).

    Shape: (4, 10, 1)
    """
    return torch.randn(batch_size, n_queries, n_coords)
