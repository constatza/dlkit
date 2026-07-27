"""GATv2 convenience presets over the conv-kind-generic embedded graph networks."""

from __future__ import annotations

from collections.abc import Callable

from torch_geometric.typing import Tensor

from dlkit.domain.nn.types import ActivationName

from .embedded import EmbeddedGraphNetwork, ScaledEmbeddedGraphNetwork
from .projection_networks import GProjection, ProjectionNetwork, ScaledGProjection

__all__ = [
    "GATv2Projection",
    "SimpleGATv2Projection",
    "ScaledGATv2Projection",
    "ScaledSimpleGATv2Projection",
    "GProjection",
    "ProjectionNetwork",
    "ScaledGProjection",
]


class GATv2Projection(EmbeddedGraphNetwork):
    """EmbeddedGraphNetwork pre-wired with a residual GATv2 message module.

    Args:
        in_channels: Number of input node feature channels.
        out_channels: Number of output node feature channels.
        hidden_size: Width of the hidden projection layers.
        num_layers: Number of GATv2 layers.
        heads: Number of attention heads per GATv2 layer.
        edge_dim: Edge feature dimensionality.
        concat: Whether to concatenate head outputs.
        dropout: Dropout probability.
        activation: Activation function applied after each layer.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        edge_dim: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize GATv2Projection.

        Args:
            in_channels: Number of input node feature channels.
            out_channels: Number of output node feature channels.
            hidden_size: Width of the hidden projection layers.
            num_layers: Number of GATv2 layers.
            heads: Number of attention heads per GATv2 layer.
            edge_dim: Edge feature dimensionality.
            concat: Whether to concatenate head outputs.
            dropout: Dropout probability.
            activation: Activation function applied after each layer.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            conv_kind="gatv2",
            residual=True,
            edge_dim=edge_dim,
            activation=activation,
            dropout=dropout,
            heads=heads,
            concat=concat,
        )


class SimpleGATv2Projection(EmbeddedGraphNetwork):
    """EmbeddedGraphNetwork pre-wired with a plain GATv2 message module.

    Args:
        in_channels: Number of input node feature channels.
        out_channels: Number of output node feature channels.
        hidden_size: Width of the hidden projection layers.
        num_layers: Number of GATv2 layers.
        heads: Number of attention heads per GATv2 layer.
        edge_dim: Edge feature dimensionality.
        concat: Whether to concatenate head outputs.
        dropout: Dropout probability.
        activation: Activation function applied after each layer.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        edge_dim: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize SimpleGATv2Projection.

        Args:
            in_channels: Number of input node feature channels.
            out_channels: Number of output node feature channels.
            hidden_size: Width of the hidden projection layers.
            num_layers: Number of GATv2 layers.
            heads: Number of attention heads per GATv2 layer.
            edge_dim: Edge feature dimensionality.
            concat: Whether to concatenate head outputs.
            dropout: Dropout probability.
            activation: Activation function applied after each layer.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            conv_kind="gatv2",
            residual=False,
            edge_dim=edge_dim,
            activation=activation,
            dropout=dropout,
            heads=heads,
            concat=concat,
        )


class ScaledGATv2Projection(ScaledEmbeddedGraphNetwork):
    """ScaledEmbeddedGraphNetwork pre-wired with a residual GATv2 message module.

    Args:
        in_channels: Number of input node feature channels.
        out_channels: Number of output node feature channels.
        hidden_size: Width of the hidden projection layers.
        num_layers: Number of GATv2 layers.
        heads: Number of attention heads per GATv2 layer.
        edge_dim: Edge feature dimensionality.
        concat: Whether to concatenate head outputs.
        dropout: Dropout probability.
        activation: Activation function applied after each layer.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        edge_dim: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize ScaledGATv2Projection.

        Args:
            in_channels: Number of input node feature channels.
            out_channels: Number of output node feature channels.
            hidden_size: Width of the hidden projection layers.
            num_layers: Number of GATv2 layers.
            heads: Number of attention heads per GATv2 layer.
            edge_dim: Edge feature dimensionality.
            concat: Whether to concatenate head outputs.
            dropout: Dropout probability.
            activation: Activation function applied after each layer.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            conv_kind="gatv2",
            residual=True,
            edge_dim=edge_dim,
            activation=activation,
            dropout=dropout,
            heads=heads,
            concat=concat,
        )


class ScaledSimpleGATv2Projection(ScaledEmbeddedGraphNetwork):
    """ScaledEmbeddedGraphNetwork pre-wired with a plain GATv2 message module.

    Args:
        in_channels: Number of input node feature channels.
        out_channels: Number of output node feature channels.
        hidden_size: Width of the hidden projection layers.
        num_layers: Number of GATv2 layers.
        heads: Number of attention heads per GATv2 layer.
        edge_dim: Edge feature dimensionality.
        concat: Whether to concatenate head outputs.
        dropout: Dropout probability.
        activation: Activation function applied after each layer.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        edge_dim: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize ScaledSimpleGATv2Projection.

        Args:
            in_channels: Number of input node feature channels.
            out_channels: Number of output node feature channels.
            hidden_size: Width of the hidden projection layers.
            num_layers: Number of GATv2 layers.
            heads: Number of attention heads per GATv2 layer.
            edge_dim: Edge feature dimensionality.
            concat: Whether to concatenate head outputs.
            dropout: Dropout probability.
            activation: Activation function applied after each layer.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            conv_kind="gatv2",
            residual=False,
            edge_dim=edge_dim,
            activation=activation,
            dropout=dropout,
            heads=heads,
            concat=concat,
        )
