from collections.abc import Callable
from torch import nn, Tensor
from torch_geometric.nn.conv import GATv2Conv


class GATv2Message(nn.Module):
    """
    Stacked Graph Attention v2 (GATv2) message-passing module.

    Args:
        hidden_size: Dimension of node feature embeddings.
        num_layers: Number of GATv2 layers to apply.
        heads: Number of attention heads per GATv2 layer.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        residual: bool = True,
        edge_dim: int = None,
        concat: bool = True,
        activation: Callable = nn.functional.elu,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_size,
                out_channels=hidden_size // heads,
                heads=heads,
                concat=concat,
                edge_dim=edge_dim,
                residual=residual,
            )
            for _ in range(num_layers)
        ])

        self.activation = activation

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor = None,
    ) -> Tensor:
        """
        Apply GATv2Conv layers sequentially with optional edge weights.

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
