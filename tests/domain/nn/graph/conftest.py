"""Fixtures for graph neural network tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def edge_index() -> torch.Tensor:
    """A tiny 3-node, 4-directed-edge graph: 0<->1, 1<->2."""
    return torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)


@pytest.fixture
def edge_attr_1d(edge_index: torch.Tensor) -> torch.Tensor:
    """Scalar edge feature per edge, shape (num_edges, 1)."""
    return torch.randn(edge_index.shape[1], 1)


@pytest.fixture
def node_features() -> torch.Tensor:
    """3 nodes, 3 input channels."""
    return torch.randn(3, 3)
