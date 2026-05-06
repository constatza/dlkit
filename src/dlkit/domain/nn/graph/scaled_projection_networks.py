"""Graph projection networks with feature scaling."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch_geometric.typing import Tensor

from dlkit.common.shapes import ShapeSpecProtocol

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


class GATv2Projection(GProjection):
    """GProjection pre-wired with a residual GATv2Message."""

    def __init__(
        self,
        *,
        unified_shape: ShapeSpecProtocol,
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        edge_dim: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        activation: Callable = nn.functional.relu,
    ) -> None:
        super().__init__(
            unified_shape=unified_shape,
            hidden_size=hidden_size,
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
    """GProjection pre-wired with a plain GATv2Message."""

    def __init__(
        self,
        *,
        unified_shape: ShapeSpecProtocol,
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        edge_dim: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        activation: Callable = nn.functional.relu,
    ) -> None:
        super().__init__(
            unified_shape=unified_shape,
            hidden_size=hidden_size,
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
    """ScaledGProjection pre-wired with a residual GATv2Message."""

    def __init__(
        self,
        *,
        unified_shape: ShapeSpecProtocol,
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        edge_dim: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        activation: Callable = nn.functional.relu,
    ) -> None:
        super().__init__(
            unified_shape=unified_shape,
            hidden_size=hidden_size,
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
    """ScaledGProjection pre-wired with a plain GATv2Message."""

    def __init__(
        self,
        *,
        unified_shape: ShapeSpecProtocol,
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        edge_dim: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        activation: Callable = nn.functional.relu,
    ) -> None:
        super().__init__(
            unified_shape=unified_shape,
            hidden_size=hidden_size,
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
