from dlkit.core.models.nn.base import ShapeAwareModel
from torch import Tensor, nn
from torch_geometric.data import Data as GraphData
from abc import abstractmethod
from dlkit.core.shape_specs import IShapeSpec


class BaseGraphNetwork(ShapeAwareModel, nn.Module):
    """Abstract base for graph neural networks (PyG-based).

    Args:
        shape_spec: IShapeSpec containing graph-specific shape information.
                   For graph networks, shapes typically include:
                   - 'x': node feature dimensions (batch-free)
                   - 'edge_index': edge connectivity shape
                   - 'edge_attr': edge attribute dimensions (if present)
                   - 'y': output dimensions
        **kwargs: Additional model-specific parameters
    """

    def __init__(self, *, unified_shape: IShapeSpec, **kwargs):
        """Initialize BaseGraphNetwork with shape specification.

        Args:
            unified_shape: IShapeSpec with graph shape information
            **kwargs: Additional parameters passed to subclasses
        """
        # Call ShapeAwareModel constructor with unified_shape
        super().__init__(unified_shape=unified_shape, **kwargs)

    def get_node_feature_dim(self) -> int | None:
        """Get the node feature dimension from shape spec.

        Returns:
            Node feature dimension or None if not available
        """
        shape_spec = self.get_unified_shape()

        # For graphs, node features are typically stored as 'x'
        node_shape = shape_spec.get_shape("x")
        return node_shape[-1] if node_shape else None

    def get_edge_feature_dim(self) -> int | None:
        """Get the edge feature dimension from shape spec.

        Returns:
            Edge feature dimension or None if not available
        """
        shape_spec = self.get_unified_shape()

        # Edge features are typically stored as 'edge_attr'
        edge_shape = shape_spec.get_shape("edge_attr")
        return edge_shape[-1] if edge_shape else None

    def accepts_shape(self, shape_spec: IShapeSpec) -> bool:
        """Check if this BaseGraphNetwork can accept the given shape specification."""
        # Graph-specific validation: we typically need node features
        node_shape = shape_spec.get_shape("x")
        if node_shape is None:
            return False

        # Node features should be positive dimension
        if len(node_shape) == 0 or node_shape[-1] <= 0:
            return False

        # Validate edge features if present
        edge_shape = shape_spec.get_shape("edge_attr")
        if edge_shape is not None:
            if len(edge_shape) == 0 or edge_shape[-1] <= 0:
                return False

        return True

    @abstractmethod
    def forward(self, x: GraphData) -> Tensor:
        """Take a PyG `Data` object → return output tensor.

        Args:
            x: PyG GraphData object containing node features, edge indices, etc.

        Returns:
            Output tensor from graph processing.
        """
        ...
