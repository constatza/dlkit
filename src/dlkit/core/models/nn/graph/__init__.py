from .gat import GATv2Message
from .projection_networks import GProjection, ProjectionNetwork
from .scaled_projection_networks import (
    GATv2Projection,
    ScaledGATv2Projection,
    ScaledGProjection,
)
from .projections import LinearProjection, SkipProjection, StackedProjection

# Modern wrapper (recommended) - uses ProcessingLightningWrapper architecture
from dlkit.core.models.wrappers import GraphLightningWrapper as GraphNetwork

__all__ = [
    # Graph models
    "ProjectionNetwork",
    "GProjection",
    "GATv2Projection",
    "ScaledGProjection",
    "ScaledGATv2Projection",
    "GATv2Message",
    "LinearProjection",
    "StackedProjection",
    "SkipProjection",
    # Wrapper
    "GraphNetwork",
]
