from collections.abc import Callable
from torch import nn
from torch_geometric.data import Data
from torch_geometric.typing import Tensor
from torch_geometric import nn as gnn
from dlkit.datatypes.dataset import Shape
from .gat import GATv2Message


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
        shape: Shape,
        hidden_size: int,
        message_module: nn.Module,
    ):
        super().__init__()
        # Assume each node has a single scalar feature
        self._in_proj = gnn.Linear(-1, hidden_size)
        self._message_module = message_module
        self._out_proj = gnn.Linear(hidden_size, 1)

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
        x = self._in_proj(data.x)

        # 2) Message passing: applies equivariant propagation
        x = self._message_module(
            x,
            data.edge_index,
            edge_attr=getattr(data, "edge_attr", None),
        )

        # 3) Output projection: (num_nodes, hidden_size) -> (num_nodes, 1)
        out = self._out_proj(x)
        return out


class GATv2Projection(GProjection):
    """
    A Graph Neural Network for nodes with scalar features that projects inputs,
    applies message passing layers, and outputs a scalar per node.
    """

    def __init__(
        self,
        *,
        shape: Shape,
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        residual: bool = True,
        edge_dim: int = None,
        concat: bool = True,
        activation: Callable = nn.functional.elu,
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
            ),
        )
