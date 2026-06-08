"""Projection modules for graph neural networks.

These building blocks are small projection heads that can be composed
within larger graph models. They intentionally avoid graph-specific
concerns so they can be reused across different network variants.
"""

from __future__ import annotations

from torch import nn
from torch_geometric import nn as gnn

from dlkit.domain.nn.primitives.skip import SkipConnection

__all__ = [
    "LinearProjection",
    "SkipProjection",
    "StackedProjection",
]


class LinearProjection(nn.Module):
    """Single linear projection."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True) -> None:
        super().__init__()
        self._linear = gnn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self._linear(x)


class StackedProjection(nn.Module):
    """Sequential stack of linear layers with activations.

    Args:
        in_features (int): Input feature count.
        hidden_features (int): Hidden layer width.
        out_features (int): Output feature count.
        num_layers (int): Number of hidden layers. Defaults to 2.
        activation (type[nn.Module]): Activation class instantiated per layer. Defaults to nn.ReLU.
        bias (bool): Whether linear layers use bias. Defaults to True.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        num_layers: int = 2,
        activation: type[nn.Module] = nn.ReLU,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError(f"num_layers must be >= 0, received {num_layers}")

        self._layers = nn.ModuleList()
        self._activations = nn.ModuleList()

        prev_features = in_features
        for _ in range(num_layers):
            self._layers.append(gnn.Linear(prev_features, hidden_features, bias=bias))
            self._activations.append(activation())
            prev_features = hidden_features

        self._output = gnn.Linear(prev_features, out_features, bias=bias)

    def forward(self, x):
        for linear, act in zip(self._layers, self._activations, strict=True):
            x = act(linear(x))
        return self._output(x)


class SkipProjection(nn.Module):
    """Stacked projection with residual (skip) connections.

    Implements the same ``y = module(x) + skip_layer(x)`` invariant as
    :class:`~dlkit.domain.nn.primitives.skip.SkipConnection`. Each residual
    block wraps exactly one linear layer plus one activation (one atomic unit),
    following He et al. (2016) identity-mapping residual design.

    The input projection has no skip path (dimension change: in -> hidden).
    Each residual block uses ``nn.Identity`` as the skip layer (hidden -> hidden).

    Args:
        in_features (int): Input feature count.
        hidden_features (int): Hidden layer width.
        out_features (int): Output feature count.
        num_layers (int): Number of residual blocks. Defaults to 2.
        activation (type[nn.Module]): Activation class instantiated per block. Defaults to nn.ReLU.
        bias (bool): Whether linear layers use bias. Defaults to True.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        num_layers: int = 2,
        activation: type[nn.Module] = nn.ReLU,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError(f"num_layers must be >= 0, received {num_layers}")

        self._input = nn.Sequential(
            gnn.Linear(in_features, hidden_features, bias=bias),
            activation(),
        )
        self._residual_blocks = nn.ModuleList(
            [
                SkipConnection(
                    nn.Sequential(
                        gnn.Linear(hidden_features, hidden_features, bias=bias),
                        activation(),
                    ),
                    skip_layer=nn.Identity(),
                )
                for _ in range(num_layers)
            ]
        )
        self._output = gnn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self._input(x)
        for block in self._residual_blocks:
            x = block(x)
        return self._output(x)
