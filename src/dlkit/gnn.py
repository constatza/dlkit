"""User-facing graph neural network namespace."""

from dlkit.domain.nn.graph import (
    EmbeddedGraphNetwork,
    EmbeddedHyperGraphNetwork,
    EmbeddedMoEGraphNetwork,
    GATv2Message,
    GATv2Projection,
    GraphConvKind,
    GraphMessage,
    ScaledEmbeddedGraphNetwork,
    ScaledGATv2Projection,
    ScaledSimpleGATv2Projection,
    SimpleGATv2Message,
    SimpleGATv2Projection,
    SimpleGraphMessage,
    resolve_graph_conv_factory,
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
    "GraphConvKind",
    "resolve_graph_conv_factory",
    "GraphMessage",
    "SimpleGraphMessage",
    "EmbeddedGraphNetwork",
    "ScaledEmbeddedGraphNetwork",
    "EmbeddedHyperGraphNetwork",
    "EmbeddedMoEGraphNetwork",
    "GraphHyperConnection",
    "GraphHyperSequential",
    "SparseMoE",
    "TopKRouter",
]
