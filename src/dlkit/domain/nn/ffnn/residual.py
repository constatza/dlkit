from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

from torch import Tensor, nn

from dlkit.domain.nn.ffnn.constrained import _resolve_hidden_size
from dlkit.domain.nn.primitives import DenseBlock, SkipConnection

if TYPE_CHECKING:
    from dlkit.common.shapes import ShapeSummary


class FeedForwardNN(nn.Module):
    """Feed-forward neural network with residual skip connections."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        layers: Sequence[int],
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = len(layers)
        self.activation = activation

        self.layers = nn.ModuleList()
        self.embedding_layer = nn.Linear(in_features, layers[0], bias=bias)

        for i in range(self.num_layers - 1):
            self.layers.append(
                SkipConnection(
                    DenseBlock(
                        layers[i],
                        layers[i + 1],
                        activation=activation,
                        normalize=normalize,
                        dropout=dropout,
                        bias=bias,
                    ),
                    layer_type="linear",
                    bias=bias,
                )
            )

        self.regression_layer = nn.Linear(layers[-1], out_features, bias=bias)

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> FeedForwardNN:
        """Build the network from a dataset-derived flat shape summary."""
        return cls(
            in_features=shape.in_features,
            out_features=shape.out_features,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding_layer(x)
        for layer in self.layers:
            x = layer(x)
        return self.regression_layer(x)


class ConstantWidthFFNN(FeedForwardNN):
    """Residual feed-forward network with constant-width hidden layers.

    Note:
        ``num_layers`` is the number of hidden width entries passed to
        :class:`FeedForwardNN`.  The number of nonlinear ``DenseBlock`` layers
        created is ``num_layers - 1`` (transitions between adjacent entries).
        With ``num_layers=1`` the network is an embedding linear followed
        immediately by the regression linear — no nonlinearity.  Use
        ``num_layers >= 2`` for a genuinely nonlinear network.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        hidden_size = _resolve_hidden_size(hidden_size, in_features, out_features)
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            layers=[hidden_size] * num_layers,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
            bias=bias,
        )
