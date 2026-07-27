"""Pluggable graph-convolution Strategy layer.

Provides a small Strategy + Adapter abstraction over the PyTorch Geometric
convolution operators used for message passing: each supported kind
(``"gatv2"``, ``"gcn"``, ``"sage"``) is normalized to the same call contract
``forward(x, edge_index, edge_attr=None) -> Tensor`` via a thin adapter
module, so the rest of the graph package (message stacks, projection
networks) never needs to know which aggregation algorithm is in use.

Adding a new conv kind means adding one adapter class and one ``match`` arm
in :func:`resolve_graph_conv_factory` -- nothing else in the graph package
changes (open/closed).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from torch import Tensor, nn
from torch_geometric.nn.conv import GATv2Conv, GCNConv, SAGEConv

from dlkit.domain.nn.types import ActivationName
from dlkit.domain.nn.utils import resolve_activation

type GraphConvKind = Literal["gatv2", "gcn", "sage"]

__all__ = [
    "GraphConvKind",
    "resolve_graph_conv_factory",
    "GraphMessage",
    "SimpleGraphMessage",
    "GATv2Message",
    "SimpleGATv2Message",
]


class _GATv2ConvAdapter(nn.Module):
    """Adapter exposing ``GATv2Conv`` via the uniform graph-conv contract.

    ``GATv2Conv`` already implements a learned residual projection
    internally when ``residual=True`` (``self.res``), so this adapter passes
    ``residual`` straight through and does not add anything of its own on
    top of the wrapped conv's output.

    Args:
        in_channels: Number of input node feature channels.
        out_channels: Total number of output node feature channels across
            all attention heads.
        heads: Number of attention heads.
        concat: Whether per-head outputs are concatenated (``True``) or
            averaged (``False``).
        edge_dim: Edge feature dimensionality; ``None`` if no edge features.
        residual: Whether ``GATv2Conv`` applies its own learned residual
            projection.
        dropout: Attention-coefficient dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        heads: int = 1,
        concat: bool = True,
        edge_dim: int | None = None,
        residual: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if concat and out_channels % heads != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by heads "
                f"({heads}) when concat=True"
            )
        per_head_out = out_channels // heads if concat else out_channels
        self.conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=per_head_out,
            heads=heads,
            concat=concat,
            edge_dim=edge_dim,
            residual=residual,
            dropout=dropout,
        )

    @property
    def res(self) -> nn.Module | None:
        """Return the wrapped conv's own learned residual projection, if any.

        Returns:
            nn.Module | None: The residual projection module, or ``None``
                when ``residual=False`` was passed at construction time.
        """
        return self.conv.res

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        """Apply the wrapped ``GATv2Conv``.

        Args:
            x: Node feature tensor of shape ``(num_nodes, in_channels)``.
            edge_index: Edge indices tensor of shape ``(2, num_edges)``.
            edge_attr: Optional edge features, passed straight through.

        Returns:
            Tensor: Updated node embeddings of shape
                ``(num_nodes, out_channels)``.
        """
        return self.conv(x, edge_index, edge_attr)


class _GCNConvAdapter(nn.Module):
    """Adapter exposing ``GCNConv`` via the uniform graph-conv contract.

    ``GCNConv`` only supports a single scalar edge weight, so ``edge_dim``
    is validated at construction time: ``None`` or ``1`` are the only valid
    values. ``GCNConv`` has no native ``residual`` kwarg, so residual
    connections are applied by this adapter's own ``forward()``.

    Args:
        in_channels: Number of input node feature channels.
        out_channels: Number of output node feature channels.
        edge_dim: Edge feature dimensionality; must be ``None`` or ``1``.
        residual: Whether to add ``x`` to the conv output.
        dropout: Feature dropout probability applied to the conv output.
        **conv_kwargs: Additional keyword arguments forwarded to
            ``GCNConv`` (e.g. ``improved``, ``cached``, ``add_self_loops``,
            ``normalize``, ``bias``).

    Raises:
        ValueError: If ``edge_dim`` is not ``None`` or ``1``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        edge_dim: int | None = None,
        residual: bool = False,
        dropout: float = 0.0,
        **conv_kwargs,
    ) -> None:
        super().__init__()
        if edge_dim not in (None, 1):
            raise ValueError(
                f"GCNConv only supports a scalar edge weight; edge_dim must "
                f"be None or 1, got {edge_dim!r}"
            )
        self.residual = residual
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv = GCNConv(in_channels, out_channels, **conv_kwargs)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        """Apply ``GCNConv``, translating ``edge_attr`` into ``edge_weight``.

        Args:
            x: Node feature tensor of shape ``(num_nodes, in_channels)``.
            edge_index: Edge indices tensor of shape ``(2, num_edges)``.
            edge_attr: Optional scalar edge features of shape
                ``(num_edges, 1)`` or ``(num_edges,)``.

        Returns:
            Tensor: Updated node embeddings of shape
                ``(num_nodes, out_channels)``.
        """
        edge_weight = edge_attr.squeeze(-1) if edge_attr is not None else None
        out = self.dropout(self.conv(x, edge_index, edge_weight))
        return x + out if self.residual else out


class _SAGEConvAdapter(nn.Module):
    """Adapter exposing ``SAGEConv`` via the uniform graph-conv contract.

    ``SAGEConv`` has no valid edge-feature dimensionality at all (unlike
    GCN's "only 1 is valid", SAGE's answer is "there is no valid value"), so
    ``edge_dim`` is validated at construction time and any non-``None``
    ``edge_attr`` is rejected at forward time as a defensive backstop.
    ``SAGEConv`` has no native ``residual`` kwarg, so residual connections
    are applied by this adapter's own ``forward()``.

    Args:
        in_channels: Number of input node feature channels.
        out_channels: Number of output node feature channels.
        edge_dim: Edge feature dimensionality; must be ``None``.
        residual: Whether to add ``x`` to the conv output.
        dropout: Feature dropout probability applied to the conv output.
        **conv_kwargs: Additional keyword arguments forwarded to
            ``SAGEConv`` (e.g. ``aggr``, ``normalize``, ``root_weight``,
            ``project``, ``bias``).

    Raises:
        ValueError: If ``edge_dim`` is not ``None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        edge_dim: int | None = None,
        residual: bool = False,
        dropout: float = 0.0,
        **conv_kwargs,
    ) -> None:
        super().__init__()
        if edge_dim is not None:
            raise ValueError(
                f"SAGEConv does not support edge features; edge_dim must be None, got {edge_dim!r}"
            )
        self.residual = residual
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv = SAGEConv(in_channels, out_channels, **conv_kwargs)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        """Apply ``SAGEConv``, rejecting any supplied edge features.

        Args:
            x: Node feature tensor of shape ``(num_nodes, in_channels)``.
            edge_index: Edge indices tensor of shape ``(2, num_edges)``.
            edge_attr: Must be ``None``; present only to satisfy the shared
                graph-conv call contract.

        Returns:
            Tensor: Updated node embeddings of shape
                ``(num_nodes, out_channels)``.

        Raises:
            ValueError: If ``edge_attr`` is not ``None``.
        """
        if edge_attr is not None:
            raise ValueError(
                "SAGEConv does not support edge features; got a non-None "
                "edge_attr at forward time despite edge_dim=None"
            )
        out = self.dropout(self.conv(x, edge_index))
        return x + out if self.residual else out


def resolve_graph_conv_factory(
    kind: GraphConvKind,
    *,
    residual: bool = False,
    edge_dim: int | None = None,
    dropout: float = 0.0,
    **conv_kwargs,
) -> Callable[[int, int], nn.Module]:
    """Resolve a ``(in_channels, out_channels) -> nn.Module`` factory for a conv kind.

    Args:
        kind: Graph convolution strategy to build a factory for.
        residual: Whether the built conv module applies a residual
            connection (mechanism differs per kind: ``GATv2Conv`` has a
            native ``residual`` kwarg; GCN/SAGE apply it in the adapter).
        edge_dim: Edge feature dimensionality; ``None`` if no edge features.
            Validity is kind-specific and enforced at construction time.
        dropout: Dropout probability; semantics are kind-specific (attention
            dropout for GATv2, feature dropout for GCN/SAGE).
        **conv_kwargs: Kind-specific knobs forwarded to the adapter (e.g.
            ``heads``/``concat`` for GATv2, ``aggr`` for SAGE).

    Returns:
        Callable[[int, int], nn.Module]: A factory that builds one adapter
        module given ``(in_channels, out_channels)``.

    Raises:
        ValueError: If ``kind`` is not one of the supported values.
    """
    match kind:
        case "gatv2":
            return lambda in_channels, out_channels: _GATv2ConvAdapter(
                in_channels,
                out_channels,
                edge_dim=edge_dim,
                residual=residual,
                dropout=dropout,
                **conv_kwargs,
            )
        case "gcn":
            return lambda in_channels, out_channels: _GCNConvAdapter(
                in_channels,
                out_channels,
                edge_dim=edge_dim,
                residual=residual,
                dropout=dropout,
                **conv_kwargs,
            )
        case "sage":
            return lambda in_channels, out_channels: _SAGEConvAdapter(
                in_channels,
                out_channels,
                edge_dim=edge_dim,
                residual=residual,
                dropout=dropout,
                **conv_kwargs,
            )
        case _:
            supported = ("gatv2", "gcn", "sage")
            raise ValueError(f"Unsupported graph conv kind {kind!r}; supported: {supported}")


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
