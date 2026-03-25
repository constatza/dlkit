"""Projection modules for graph neural networks.

These building blocks are small projection heads that can be composed
within larger graph models. They intentionally avoid graph-specific
concerns so they can be reused across different network variants.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

from torch import nn
from torch_geometric import nn as gnn

ActivationFactory = Callable[[], nn.Module] | type[nn.Module] | nn.Module

__all__ = [
    "LinearProjection",
    "SkipProjection",
    "StackedProjection",
]


def _build_activation(factory: ActivationFactory | None) -> nn.Module:
    """Instantiate an activation module from a factory-like input."""
    if factory is None:
        return nn.GELU()

    if isinstance(factory, nn.Module):
        return deepcopy(factory)

    if isinstance(factory, type) and issubclass(factory, nn.Module):
        return factory()

    if callable(factory):
        result = factory()
        if not isinstance(result, nn.Module):
            raise TypeError("Activation factory must return an nn.Module instance.")
        return result

    raise TypeError(f"Unsupported activation factory type: {type(factory)!r}")


class LinearProjection(nn.Module):
    """Single linear projection."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True) -> None:
        super().__init__()
        self._linear = gnn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self._linear(x)


class StackedProjection(nn.Module):
    """Sequential stack of linear layers with activations."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        num_layers: int = 2,
        activation: ActivationFactory | None = None,
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
            self._activations.append(_build_activation(activation))
            prev_features = hidden_features

        self._output = gnn.Linear(prev_features, out_features, bias=bias)

    def forward(self, x):
        for linear, activation in zip(self._layers, self._activations, strict=True):
            x = activation(linear(x))
        return self._output(x)


class SkipProjection(nn.Module):
    """Stacked projection with residual (skip) connections."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        num_layers: int = 2,
        activation: ActivationFactory | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError(f"num_layers must be >= 0, received {num_layers}")

        self._input = gnn.Linear(in_features, hidden_features, bias=bias)
        self._input_activation = _build_activation(activation)

        self._residual_layers = nn.ModuleList(
            [gnn.Linear(hidden_features, hidden_features, bias=bias) for _ in range(num_layers)]
        )
        self._residual_activations = nn.ModuleList(
            [_build_activation(activation) for _ in range(num_layers)]
        )

        self._output = gnn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self._input_activation(self._input(x))

        for layer, activation in zip(
            self._residual_layers,
            self._residual_activations,
            strict=True,
        ):
            residual = x
            x = activation(layer(x))
            x = x + residual

        return self._output(x)
