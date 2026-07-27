"""Stacked graph message-passing modules, generic over ``conv_kind``.

Composes the adapter/factory layer in :mod:`dlkit.domain.nn.graph.conv` into
multi-layer message-passing stacks. Kept separate from ``conv.py`` so that
module stays scoped to PyTorch Geometric normalization only, mirroring how
``primitives/dense.py`` (block-strategy factory) and ``ffnn/hyper_moe.py``
(the composition layer built on top of it) are separate files elsewhere in
this package.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

from dlkit.domain.nn.types import ActivationName
from dlkit.domain.nn.utils import resolve_activation

from .conv import GraphConvKind, resolve_graph_conv_factory

__all__ = [
    "GraphMessage",
    "SimpleGraphMessage",
    "GATv2Message",
    "SimpleGATv2Message",
]


class _GraphMessageBase(nn.Module):
    """Stacked graph message-passing module, generic over ``conv_kind``.

    Args:
        conv_kind: Graph convolution strategy used for every layer.
        hidden_size: Dimension of node feature embeddings.
        num_layers: Number of graph-conv layers to apply.
        residual: Whether layers use residual connections (mechanism is
            kind-specific; see :func:`resolve_graph_conv_factory`).
        edge_dim: Optional edge feature dimensionality.
        activation: Activation function applied after each layer.
        dropout: Dropout probability; semantics are kind-specific.
        **conv_kwargs: Kind-specific knobs (e.g. ``heads``/``concat`` for
            GATv2, ``aggr`` for SAGE) forwarded to every layer.
    """

    def __init__(
        self,
        *,
        conv_kind: GraphConvKind,
        hidden_size: int,
        num_layers: int,
        residual: bool,
        edge_dim: int | None = None,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        dropout: float = 0.0,
        **conv_kwargs,
    ) -> None:
        super().__init__()
        factory = resolve_graph_conv_factory(
            conv_kind,
            residual=residual,
            edge_dim=edge_dim,
            dropout=dropout,
            **conv_kwargs,
        )
        self.layers = nn.ModuleList([factory(hidden_size, hidden_size) for _ in range(num_layers)])
        self.activation = resolve_activation(activation)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Apply the stacked conv layers sequentially.

        Args:
            x: Node feature tensor of shape ``(num_nodes, hidden_size)``.
            edge_index: Edge indices tensor of shape ``(2, num_edges)``.
            edge_attr: Optional edge features; validity is kind-specific.

        Returns:
            Tensor: Updated node embeddings of shape
                ``(num_nodes, hidden_size)``.
        """
        for conv in self.layers:
            x = self.activation(conv(x, edge_index, edge_attr))
        return x


class GraphMessage(_GraphMessageBase):
    """Stacked graph message-passing with residual connections."""

    def __init__(
        self,
        *,
        conv_kind: GraphConvKind = "gatv2",
        hidden_size: int,
        num_layers: int,
        edge_dim: int | None = None,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        dropout: float = 0.0,
        **conv_kwargs,
    ) -> None:
        super().__init__(
            conv_kind=conv_kind,
            hidden_size=hidden_size,
            num_layers=num_layers,
            residual=True,
            edge_dim=edge_dim,
            activation=activation,
            dropout=dropout,
            **conv_kwargs,
        )


class SimpleGraphMessage(_GraphMessageBase):
    """Stacked graph message-passing without residual connections."""

    def __init__(
        self,
        *,
        conv_kind: GraphConvKind = "gatv2",
        hidden_size: int,
        num_layers: int,
        edge_dim: int | None = None,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        dropout: float = 0.0,
        **conv_kwargs,
    ) -> None:
        super().__init__(
            conv_kind=conv_kind,
            hidden_size=hidden_size,
            num_layers=num_layers,
            residual=False,
            edge_dim=edge_dim,
            activation=activation,
            dropout=dropout,
            **conv_kwargs,
        )


class GATv2Message(GraphMessage):
    """Stacked GATv2 message-passing with residual connections.

    A one-line preset over :class:`GraphMessage` fixing ``conv_kind="gatv2"``,
    kept for backward compatibility with code that imports ``GATv2Message``
    directly.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        edge_dim: int | None = None,
        concat: bool = True,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            conv_kind="gatv2",
            hidden_size=hidden_size,
            num_layers=num_layers,
            edge_dim=edge_dim,
            activation=activation,
            dropout=dropout,
            heads=heads,
            concat=concat,
        )


class SimpleGATv2Message(SimpleGraphMessage):
    """Stacked GATv2 message-passing without residual connections.

    A one-line preset over :class:`SimpleGraphMessage` fixing
    ``conv_kind="gatv2"``, kept for backward compatibility with code that
    imports ``SimpleGATv2Message`` directly.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        edge_dim: int | None = None,
        concat: bool = True,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            conv_kind="gatv2",
            hidden_size=hidden_size,
            num_layers=num_layers,
            edge_dim=edge_dim,
            activation=activation,
            dropout=dropout,
            heads=heads,
            concat=concat,
        )
