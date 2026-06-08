"""Graph projection networks built from projection modules."""

from __future__ import annotations

from torch import nn
from torch_geometric.typing import Tensor

from .base import BaseGraphNetwork
from .projections import SkipProjection

__all__ = [
    "GProjection",
    "ProjectionNetwork",
]


class ProjectionNetwork(BaseGraphNetwork):
    """Common projection network foundation without normalization.

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
        hidden_size: int,
        edge_dim: int | None = None,
        message_module: nn.Module | None = None,
        input_projection: nn.Module | None = None,
        output_projection: nn.Module | None = None,
    ):
        """Initialize ProjectionNetwork.

        Args:
            in_channels: Number of input node feature channels.
            out_channels: Number of output node feature channels.
            hidden_size: Width of the hidden projection layers.
            edge_dim: Edge feature dimensionality; ``None`` if no edge features.
            message_module: Optional message-passing module.
            input_projection: Optional custom input projection module.
            output_projection: Optional custom output projection module.
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels, edge_dim=edge_dim)

        activation = nn.ReLU
        self._in_proj = input_projection or SkipProjection(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            num_layers=2,
            activation=activation,
        )
        self._out_proj = output_projection or SkipProjection(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels,
            num_layers=2,
            activation=activation,
        )

        self._message_module = message_module

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through input projection, message passing, and output projection.

        Args:
            x: Node feature tensor.
            edge_index: Edge connectivity tensor (2 × num_edges).
            edge_attr: Optional edge attribute tensor.

        Returns:
            Output tensor from graph processing.
        """
        x = self._in_proj(x)
        x = self._apply_message_module(x, edge_index, edge_attr)
        return self._out_proj(x)

    def _apply_message_module(
        self,
        features: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        if self._message_module is None:
            return features
        return self._message_module(features, edge_index, edge_attr=edge_attr)


class GProjection(ProjectionNetwork):
    """Projection network without feature scaling.

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
        """Initialize GProjection.

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
