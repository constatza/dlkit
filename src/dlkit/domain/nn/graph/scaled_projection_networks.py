"""Graph projection networks with feature scaling."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch_geometric.typing import Tensor

from dlkit.shared.shapes import ShapeSpecProtocol

from .gat import GATv2Message
from .projection_networks import GProjection, ProjectionNetwork

EPSILON = 1e-14

__all__ = [
    "GATv2Projection",
    "ScaledGATv2Projection",
    "ScaledGProjection",
]


class ScaledGProjection(ProjectionNetwork):
    """Projection network with column-wise input scaling."""

    def __init__(
        self,
        *,
        unified_shape: ShapeSpecProtocol,
        hidden_size: int = 64,
        message_module: nn.Module | None = None,
        input_projection: nn.Module | None = None,
        output_projection: nn.Module | None = None,
    ):
        super().__init__(
            unified_shape=unified_shape,
            hidden_size=hidden_size,
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


def GATv2Projection(
    *,
    unified_shape: ShapeSpecProtocol,
    hidden_size: int,
    num_layers: int,
    heads: int = 1,
    residual: bool = True,
    edge_dim: int = 1,
    concat: bool = True,
    dropout: float = 0.0,
    activation: Callable = nn.functional.relu,
) -> GProjection:
    """Factory: GProjection with a GATv2Message module pre-wired.

    Args:
        unified_shape: Shape specification for graph inputs/outputs.
        hidden_size: Dimension of node feature embeddings.
        num_layers: Number of GATv2 layers.
        heads: Number of attention heads.
        residual: Whether to use residual connections in GAT layers.
        edge_dim: Edge feature dimension.
        concat: Whether to concatenate head outputs.
        dropout: Dropout probability.
        activation: Activation function for GAT layers.

    Returns:
        GProjection with a GATv2Message as message_module.
    """
    return GProjection(
        unified_shape=unified_shape,
        hidden_size=hidden_size,
        message_module=GATv2Message(
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


def ScaledGATv2Projection(
    *,
    unified_shape: ShapeSpecProtocol,
    hidden_size: int,
    num_layers: int,
    heads: int = 1,
    residual: bool = True,
    edge_dim: int = 1,
    concat: bool = True,
    dropout: float = 0.0,
    activation: Callable = nn.functional.relu,
) -> ScaledGProjection:
    """Factory: ScaledGProjection with a GATv2Message module pre-wired.

    Args:
        unified_shape: Shape specification for graph inputs/outputs.
        hidden_size: Dimension of node feature embeddings.
        num_layers: Number of GATv2 layers.
        heads: Number of attention heads.
        residual: Whether to use residual connections in GAT layers.
        edge_dim: Edge feature dimension.
        concat: Whether to concatenate head outputs.
        dropout: Dropout probability.
        activation: Activation function for GAT layers.

    Returns:
        ScaledGProjection with a GATv2Message as message_module.
    """
    return ScaledGProjection(
        unified_shape=unified_shape,
        hidden_size=hidden_size,
        message_module=GATv2Message(
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
