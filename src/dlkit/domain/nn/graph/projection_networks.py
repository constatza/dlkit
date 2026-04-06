"""Graph projection networks built from projection modules."""

from __future__ import annotations

from torch import nn
from torch_geometric.typing import Tensor

from dlkit.common.shapes import ShapeSpecProtocol

from .base import BaseGraphNetwork
from .projections import SkipProjection

__all__ = [
    "GProjection",
    "ProjectionNetwork",
]


class ProjectionNetwork(BaseGraphNetwork):
    """Common projection network foundation without normalization."""

    def __init__(
        self,
        *,
        unified_shape: ShapeSpecProtocol,
        hidden_size: int,
        message_module: nn.Module | None = None,
        input_projection: nn.Module | None = None,
        output_projection: nn.Module | None = None,
    ):
        super().__init__(unified_shape=unified_shape)

        in_dim = self.get_node_feature_dim()
        if in_dim is None:
            raise ValueError("Shape spec must contain 'x' (node features) dimension")

        out_shape = unified_shape.get_shape("y")
        out_dim = out_shape[-1] if out_shape else 1

        activation = nn.GELU
        self._in_proj = input_projection or SkipProjection(
            in_features=in_dim,
            hidden_features=hidden_size,
            out_features=hidden_size,
            num_layers=2,
            activation=activation,
        )
        self._out_proj = output_projection or SkipProjection(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_dim,
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
    """Projection network without feature scaling."""

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
