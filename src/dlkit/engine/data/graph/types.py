"""Graph-specific data types.

This module re-exports PyTorch Geometric types for convenience and defines
dlkit-specific type aliases for graph processing workflows.
"""

# Re-export PyG core types for convenience and discoverability
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.typing import OptTensor, PairTensor, Tensor

# dlkit-specific type aliases
type GraphDict = dict[str, Tensor]
"""Flattened Data/Batch representation used in dlkit processing pipelines.
Converts PyG Data/Batch objects to dict format for pipeline compatibility."""

type GraphInput = Data | Batch | GraphDict
"""All accepted input formats in graph wrappers.
Supports PyG Data (single graph), Batch (batched graphs), or dict (pipeline format)."""

__all__ = [
    # PyG types (re-exported for convenience)
    "Batch",
    "Data",
    "InMemoryDataset",
    "OptTensor",
    "PairTensor",
    "Tensor",
    # dlkit-specific type aliases
    "GraphDict",
    "GraphInput",
]
