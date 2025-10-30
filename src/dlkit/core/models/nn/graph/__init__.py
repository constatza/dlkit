from .gat import GATv2Message
from .projection_networks import GATv2Projection, GProjection
from .scaled_projection_networks import ScaledGATv2Projection, ScaledGProjection
from .projections import LinearProjection, SkipProjection, StackedProjection

# Modern wrapper (recommended) - uses ProcessingLightningWrapper architecture
from dlkit.core.models.wrappers import GraphLightningWrapper as GraphNetwork

# Legacy wrapper (deprecated) - for backwards compatibility only
from .wrap import LegacyGraphWrapper

__all__ = [
    # Graph models
    "GProjection",
    "GATv2Projection",
    "ScaledGProjection",
    "ScaledGATv2Projection",
    "GATv2Message",
    "LinearProjection",
    "StackedProjection",
    "SkipProjection",
    # Wrappers
    "GraphNetwork",  # Modern wrapper (GraphLightningWrapper)
    "LegacyGraphWrapper",  # Deprecated - use GraphNetwork instead
]
