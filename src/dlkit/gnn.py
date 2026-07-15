"""User-facing graph neural network namespace."""

from dlkit.domain.nn.graph import (
    GATv2Message,
    GATv2Projection,
    ScaledGATv2Projection,
    ScaledSimpleGATv2Projection,
    SimpleGATv2Message,
    SimpleGATv2Projection,
)
from dlkit.domain.nn.primitives import (
    GraphHyperConnection,
    GraphHyperSequential,
    SparseMoE,
    TopKRouter,
)

__all__ = [
    "GATv2Message",
    "SimpleGATv2Message",
    "GATv2Projection",
    "SimpleGATv2Projection",
    "ScaledGATv2Projection",
    "ScaledSimpleGATv2Projection",
    "GraphHyperConnection",
    "GraphHyperSequential",
    "SparseMoE",
    "TopKRouter",
]
