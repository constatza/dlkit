from collections.abc import Callable

from torch import Tensor, nn
from torch_geometric.nn.conv import GATv2Conv


class _GATv2MessageBase(nn.Module):
    """Stacked Graph Attention v2 (GATv2) message-passing module.

    Args:
        hidden_size: Dimension of node feature embeddings.
        num_layers: Number of GATv2 layers to apply.
        heads: Number of attention heads per GATv2 layer.
        _residual: Whether to use residual connections in GAT layers.
        edge_dim: Optional edge feature dimension.
        concat: Whether to concatenate head outputs.
        activation: Activation function applied after each layer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        _residual: bool = True,
        edge_dim: int | None = None,
        concat: bool = True,
        activation: Callable = nn.functional.relu,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=hidden_size,
                    out_channels=hidden_size // heads,
                    heads=heads,
                    concat=concat,
                    edge_dim=edge_dim,
                    residual=_residual,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.activation = activation

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Apply GATv2Conv layers sequentially with optional edge weights.

        Args:
            x: Node feature tensor of shape (num_nodes, hidden_size).
            edge_index: Edge indices tensor of shape (2, num_edges).
            edge_attr: Optional edge features, unused by GATv2.

        Returns:
            Tensor of shape (num_nodes, hidden_size): updated node embeddings.
        """
        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = self.activation(x)
        return x


class GATv2Message(_GATv2MessageBase):
    """Stacked GATv2 message-passing with residual connections."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        edge_dim: int | None = None,
        concat: bool = True,
        activation: Callable = nn.functional.relu,
        dropout: float = 0.0,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_layers=num_layers,
            heads=heads,
            _residual=True,
            edge_dim=edge_dim,
            concat=concat,
            activation=activation,
            dropout=dropout,
        )


class SimpleGATv2Message(_GATv2MessageBase):
    """Stacked GATv2 message-passing without residual connections."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        edge_dim: int | None = None,
        concat: bool = True,
        activation: Callable = nn.functional.relu,
        dropout: float = 0.0,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_layers=num_layers,
            heads=heads,
            _residual=False,
            edge_dim=edge_dim,
            concat=concat,
            activation=activation,
            dropout=dropout,
        )
