import torch
from collections.abc import Callable
from torch import nn
from torch_geometric.data import Data
from torch_geometric.typing import Tensor
from torch_geometric import nn as gnn
# from dlkit.core.datatypes.dataset import Shape  # Removed - using IShapeSpec
from dlkit.core.shape_specs import IShapeSpec
from .gat import GATv2Message

EPSILON = 1e-14


class GProjection(nn.Module):
    """
    A Graph Neural Network for nodes with scalar features that projects inputs,
    applies message passing layers, and outputs a scalar per node.

    Args:
        hidden_size: Dimension of the hidden node embeddings.
        message_module: An external nn.Module implementing message passing;
            its forward signature should be (x: Tensor,
                                            edge_index: Tensor,
                                            edge_attr: Optional[Tensor] = None,
                                            edge_weight: Optional[Tensor] = None) -> Tensor.
    """

    def __init__(
        self,
        shape_spec=None,  # Not used in this graph model
        hidden_size: int = 64,
        message_module: nn.Module = None,
    ):
        super().__init__()
        # Assume each node has a single scalar feature
        self._in_proj = nn.Sequential(
            gnn.Linear(1, hidden_size),
            nn.ReLU(),
            gnn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            gnn.Linear(hidden_size, hidden_size),
        )

        self._message_module = message_module
        self._out_proj = nn.Sequential(
            gnn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            gnn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            gnn.Linear(hidden_size, 1),
        )

    def forward(self, data: Data) -> Tensor:
        """
        Perform forward propagation through the NodeScalarGNN.

        Args:
            data: Data object with attributes:
                x: Tensor of shape (num_nodes, 1)
                edge_index: LongTensor of shape (2, num_edges)
                edge_attr: Optional Tensor of shape (num_edges, num_edge_features)

        Returns:
            Tensor of shape (num_nodes, 1): scalar output per node.
        """
        # 1) Input projection: (num_nodes, 1) -> (num_nodes, hidden_size)
        x = data.x
        with torch.no_grad():
            scale = x.norm(dim=0) / x.size(dim=0) ** 0.5 + EPSILON
        x = x / scale
        x = self._in_proj(x)

        edge_weight = getattr(data, "edge_weight", None)
        edge_weight = edge_weight

        # 2) Message passing: applies equivariant propagation
        x = self._message_module(
            x,
            data.edge_index,
            edge_attr=edge_weight if edge_weight is not None else data.edge_attr,
        )

        # 3) Output projection: (num_nodes, hidden_size) -> (num_nodes, 1)
        out = self._out_proj(x)
        out = out * scale
        return out


class GATv2Projection(GProjection):
    """
    A Graph Neural Network for nodes with scalar features that projects inputs,
    applies message passing layers, and outputs a scalar per node.
    """

    def __init__(
        self,
        *,
        shape_spec=None,  # Not used in GATv2Projection
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        residual: bool = True,
        edge_dim: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        activation: Callable = nn.functional.relu,
    ):
        super().__init__(
            shape,
            hidden_size,
            GATv2Message(
                hidden_size=hidden_size,
                num_layers=num_layers,
                heads=heads,
                residual=residual,
                edge_dim=edge_dim,
                concat=concat,
                activation=activation,
                dropout=dropout,
            ),
        )
