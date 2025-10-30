"""Graph projection networks built from projection modules."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn
from torch_geometric.data import Data
from torch_geometric.typing import Tensor

from dlkit.core.shape_specs import IShapeSpec
from .base import BaseGraphNetwork
from .gat import GATv2Message
from .projections import SkipProjection

__all__ = [
    "GProjection",
    "GATv2Projection",
]


class _ProjectionNetwork(BaseGraphNetwork):
    """Common projection network foundation without normalization."""

    def __init__(
        self,
        *,
        unified_shape: IShapeSpec,
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

    def forward(self, data: Data) -> Tensor:
        x = self._in_proj(data.x)
        x = self._apply_message_module(x, data)
        return self._out_proj(x)

    def _apply_message_module(self, features: Tensor, data: Data) -> Tensor:
        if self._message_module is None:
            return features

        edge_weight = getattr(data, "edge_weight", None)
        edge_attr = edge_weight if edge_weight is not None else getattr(data, "edge_attr", None)

        return self._message_module(
            features,
            data.edge_index,
            edge_attr=edge_attr,
        )


class GProjection(_ProjectionNetwork):
    """Projection network without feature scaling."""

    def __init__(
        self,
        *,
        unified_shape: IShapeSpec,
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


class GATv2Projection(GProjection):
    """GATv2-based projection network without scaling."""

    def __init__(
        self,
        *,
        unified_shape: IShapeSpec,
        hidden_size: int,
        num_layers: int,
        heads: int = 1,
        residual: bool = True,
        edge_dim: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        activation: Callable = nn.functional.relu,
    ):
        super().__init__(
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
