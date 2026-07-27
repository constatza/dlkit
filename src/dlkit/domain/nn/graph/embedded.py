"""GNN composites parameterized by graph-convolution kind.

The GNN analog of the FFNN family's ``EmbeddedFFNN``: fixed input/output
projections wrap a message-passing body whose aggregation strategy is
selected via ``conv_kind`` rather than hardcoded, so new conv kinds never
require touching these classes (open/closed).
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

from dlkit.domain.nn.types import ActivationName

from .conv import GraphConvKind, GraphMessage, SimpleGraphMessage
from .projection_networks import GProjection, ScaledGProjection

__all__ = ["EmbeddedGraphNetwork", "ScaledEmbeddedGraphNetwork"]


def _build_message_module(
    *,
    conv_kind: GraphConvKind,
    residual: bool,
    hidden_size: int,
    num_layers: int,
    edge_dim: int | None,
    activation: ActivationName | Callable[[Tensor], Tensor] | None,
    dropout: float,
    **conv_kwargs,
) -> nn.Module:
    """Build the default message-passing module for an embedded graph network.

    Args:
        conv_kind: Graph convolution strategy used for every layer.
        residual: Whether the message-passing module uses residual
            connections.
        hidden_size: Dimension of node feature embeddings.
        num_layers: Number of stacked graph-conv layers.
        edge_dim: Optional edge feature dimensionality.
        activation: Activation function applied after each layer.
        dropout: Dropout probability; semantics are kind-specific.
        **conv_kwargs: Kind-specific knobs (e.g. ``heads``/``concat`` for
            GATv2, ``aggr`` for SAGE) forwarded to every layer.

    Returns:
        nn.Module: A :class:`GraphMessage` or :class:`SimpleGraphMessage`
        instance, depending on ``residual``.
    """
    message_cls = GraphMessage if residual else SimpleGraphMessage
    return message_cls(
        conv_kind=conv_kind,
        hidden_size=hidden_size,
        num_layers=num_layers,
        edge_dim=edge_dim,
        activation=activation,
        dropout=dropout,
        **conv_kwargs,
    )


class EmbeddedGraphNetwork(GProjection):
    """GProjection pre-wired with a conv-kind-selectable message-passing module.

    Args:
        in_channels: Number of input node feature channels.
        out_channels: Number of output node feature channels.
        hidden_size: Width of the hidden projection layers and message
            module.
        num_layers: Number of stacked graph-conv layers.
        conv_kind: Graph convolution strategy used for message passing
            (``"gatv2"``, ``"gcn"``, or ``"sage"``).
        residual: Whether the message-passing module uses residual
            connections.
        edge_dim: Edge feature dimensionality; ``None`` if no edge features.
            Validity is kind-specific (see :mod:`dlkit.domain.nn.graph.conv`).
        activation: Activation function applied after each conv layer.
        dropout: Dropout probability; semantics are kind-specific.
        message_module: Optional custom message-passing module. When
            supplied, ``conv_kind``/``residual``/``num_layers``/``dropout``/
            ``**conv_kwargs`` are ignored for module construction.
        input_projection: Optional custom input projection module.
        output_projection: Optional custom output projection module.
        **conv_kwargs: Kind-specific knobs (e.g. ``heads``/``concat`` for
            GATv2, ``aggr`` for SAGE) forwarded to the underlying conv.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 64,
        num_layers: int,
        conv_kind: GraphConvKind = "gatv2",
        residual: bool = True,
        edge_dim: int | None = None,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        dropout: float = 0.0,
        message_module: nn.Module | None = None,
        input_projection: nn.Module | None = None,
        output_projection: nn.Module | None = None,
        **conv_kwargs,
    ) -> None:
        """Initialize EmbeddedGraphNetwork.

        Args:
            in_channels: Number of input node feature channels.
            out_channels: Number of output node feature channels.
            hidden_size: Width of the hidden projection layers and message
                module.
            num_layers: Number of stacked graph-conv layers.
            conv_kind: Graph convolution strategy used for message passing.
            residual: Whether the message-passing module uses residual
                connections.
            edge_dim: Edge feature dimensionality; ``None`` if no edge
                features.
            activation: Activation function applied after each conv layer.
            dropout: Dropout probability; semantics are kind-specific.
            message_module: Optional custom message-passing module.
            input_projection: Optional custom input projection module.
            output_projection: Optional custom output projection module.
            **conv_kwargs: Kind-specific knobs forwarded to the underlying
                conv.
        """
        if message_module is None:
            message_module = _build_message_module(
                conv_kind=conv_kind,
                residual=residual,
                hidden_size=hidden_size,
                num_layers=num_layers,
                edge_dim=edge_dim,
                activation=activation,
                dropout=dropout,
                **conv_kwargs,
            )
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            edge_dim=edge_dim,
            message_module=message_module,
            input_projection=input_projection,
            output_projection=output_projection,
        )


class ScaledEmbeddedGraphNetwork(ScaledGProjection):
    """ScaledGProjection pre-wired with a conv-kind-selectable message-passing module.

    Args:
        in_channels: Number of input node feature channels.
        out_channels: Number of output node feature channels.
        hidden_size: Width of the hidden projection layers and message
            module.
        num_layers: Number of stacked graph-conv layers.
        conv_kind: Graph convolution strategy used for message passing
            (``"gatv2"``, ``"gcn"``, or ``"sage"``).
        residual: Whether the message-passing module uses residual
            connections.
        edge_dim: Edge feature dimensionality; ``None`` if no edge features.
            Validity is kind-specific (see :mod:`dlkit.domain.nn.graph.conv`).
        activation: Activation function applied after each conv layer.
        dropout: Dropout probability; semantics are kind-specific.
        message_module: Optional custom message-passing module. When
            supplied, ``conv_kind``/``residual``/``num_layers``/``dropout``/
            ``**conv_kwargs`` are ignored for module construction.
        input_projection: Optional custom input projection module.
        output_projection: Optional custom output projection module.
        **conv_kwargs: Kind-specific knobs (e.g. ``heads``/``concat`` for
            GATv2, ``aggr`` for SAGE) forwarded to the underlying conv.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 64,
        num_layers: int,
        conv_kind: GraphConvKind = "gatv2",
        residual: bool = True,
        edge_dim: int | None = None,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        dropout: float = 0.0,
        message_module: nn.Module | None = None,
        input_projection: nn.Module | None = None,
        output_projection: nn.Module | None = None,
        **conv_kwargs,
    ) -> None:
        """Initialize ScaledEmbeddedGraphNetwork.

        Args:
            in_channels: Number of input node feature channels.
            out_channels: Number of output node feature channels.
            hidden_size: Width of the hidden projection layers and message
                module.
            num_layers: Number of stacked graph-conv layers.
            conv_kind: Graph convolution strategy used for message passing.
            residual: Whether the message-passing module uses residual
                connections.
            edge_dim: Edge feature dimensionality; ``None`` if no edge
                features.
            activation: Activation function applied after each conv layer.
            dropout: Dropout probability; semantics are kind-specific.
            message_module: Optional custom message-passing module.
            input_projection: Optional custom input projection module.
            output_projection: Optional custom output projection module.
            **conv_kwargs: Kind-specific knobs forwarded to the underlying
                conv.
        """
        if message_module is None:
            message_module = _build_message_module(
                conv_kind=conv_kind,
                residual=residual,
                hidden_size=hidden_size,
                num_layers=num_layers,
                edge_dim=edge_dim,
                activation=activation,
                dropout=dropout,
                **conv_kwargs,
            )
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            edge_dim=edge_dim,
            message_module=message_module,
            input_projection=input_projection,
            output_projection=output_projection,
        )
