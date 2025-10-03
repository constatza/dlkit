from .base import BaseGraphNetwork
from .projection import GProjection, GATv2Projection
from .gat import GATv2Message

# Import new graph wrapper directly
from dlkit.core.models.wrappers import GraphLightningWrapper as GraphNetwork

__all__ = ["GProjection", "GATv2Projection", "GATv2Message", "GraphNetwork", "BaseGraphNetwork"]
