"""Graph projection networks with feature scaling."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch_geometric.typing import Tensor

from .gat import GATv2Message, SimpleGATv2Message
from .projection_networks import GProjection, ProjectionNetwork

EPSILON = 1e-14

__all__ = [
    "GATv2Projection",
    "SimpleGATv2Projection",
    "ScaledGATv2Projection",
    "ScaledSimpleGATv2Projection",
    "ScaledGProjection",
]


class ScaledGProjection(ProjectionNetwork):
    """Projection network with column-wise input scaling.

    Args:
        in_channels: Number of input node feature channels.
        out_channels: Number of output node feature channels.
        hidden_size: Width of the hidden projection layers.
        edge_dim: Edge feature dimensionality; ``None`` if no edge features.
        message_module: Optional message-passing module.
        input_projection: Optional custom input projection module.
        output_projection: Optional custom output projection module.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 64,
        edge_dim: int | None = None,
        message_module: nn.Module | None = None,
        input_projection: nn.Module | None = None,
        output_projection: nn.Module | None = None,
    ):
        """Initialize ScaledGProjection.

        Args:
            in_channels: Number of input node feature channels.
            out_channels: Number of output node feature channels.
            hidden_size: Width of the hidden projection layers.
            edge_dim: Edge feature dimensionality; ``None`` if no edge features.
            message_module: Optional message-passing module.
            input_projection: Optional custom input projection module.
            output_projection: Optional custom output projection module.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            edge_dim=edge_dim,
            message_module=message_module,
            input_projection=input_projection,
            output_projection=output_projection,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Forward pass with column-wise input normalization and optional output rescaling.

        Args:
            x: Node feature tensor.
            edge_index: Edge connectivity tensor (2 × num_edges).
            edge_attr: Optional edge attribute tensor.

        Returns:
            Output tensor from graph processing.
        """
        x, scale = self._normalize_inputs(x)
        x = self._in_proj(x)
        x = self._apply_message_module(x, edge_index, edge_attr)
        out = self._out_proj(x)

        if scale is not None and out.size(-1) == scale.size(-1):
            out = out * scale

        return out

    @staticmethod
    def _normalize_inputs(x: Tensor) -> tuple[Tensor, Tensor | None]:
        if x.numel() == 0:
            return x, None

        num_nodes = x.size(0)
        if num_nodes == 0:
            return x, None

        with torch.no_grad():
            scale = x.norm(dim=0, keepdim=True) / (num_nodes**0.5)
            scale = scale.clamp_min(EPSILON)

        normalized = x / scale
        return normalized, scale


class GATv2Projection(GProjection):
    """GProjection pre-wired with a residual GATv2Message.

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
        activation: Callable = nn.functional.relu,
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
            edge_dim=edge_dim,
            message_module=GATv2Message(
                hidden_size=hidden_size,
                num_layers=num_layers,
                heads=heads,
                edge_dim=edge_dim,
                concat=concat,
                activation=activation,
                dropout=dropout,
            ),
        )


class SimpleGATv2Projection(GProjection):
    """GProjection pre-wired with a plain GATv2Message.

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
        activation: Callable = nn.functional.relu,
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
            edge_dim=edge_dim,
            message_module=SimpleGATv2Message(
                hidden_size=hidden_size,
                num_layers=num_layers,
                heads=heads,
                edge_dim=edge_dim,
                concat=concat,
                activation=activation,
                dropout=dropout,
            ),
        )


class ScaledGATv2Projection(ScaledGProjection):
    """ScaledGProjection pre-wired with a residual GATv2Message.

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
        activation: Callable = nn.functional.relu,
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
            edge_dim=edge_dim,
            message_module=GATv2Message(
                hidden_size=hidden_size,
                num_layers=num_layers,
                heads=heads,
                edge_dim=edge_dim,
                concat=concat,
                activation=activation,
                dropout=dropout,
            ),
        )


class ScaledSimpleGATv2Projection(ScaledGProjection):
    """ScaledGProjection pre-wired with a plain GATv2Message.

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
        activation: Callable = nn.functional.relu,
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
            edge_dim=edge_dim,
            message_module=SimpleGATv2Message(
                hidden_size=hidden_size,
                num_layers=num_layers,
                heads=heads,
                edge_dim=edge_dim,
                concat=concat,
                activation=activation,
                dropout=dropout,
            ),
        )
