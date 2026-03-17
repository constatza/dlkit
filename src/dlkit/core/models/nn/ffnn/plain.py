from collections.abc import Sequence, Callable
from typing import Literal

import torch.nn as nn
from torch import Tensor

from dlkit.core.models.nn.base import DLKitModel
from dlkit.core.models.nn.primitives import DenseBlock


class SimpleFeedForwardNN(DLKitModel):
    """Feed-forward neural network without skip connections.

    Mirrors FeedForwardNN but uses bare DenseBlocks instead of SkipConnection
    wrappers, giving a plain sequential architecture.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        layers: Sequence of hidden layer sizes.
        activation: Activation function to use (default: gelu).
        normalize: Type of normalization ('batch', 'layer', or None).
        dropout: Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        layers: Sequence[int],
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_layer = nn.Linear(in_features, layers[0])

        self.layers = nn.ModuleList(
            DenseBlock(
                layers[i],
                layers[i + 1],
                activation=activation,
                normalize=normalize,
                dropout=dropout,
            )
            for i in range(len(layers) - 1)
        )

        self.regression_layer = nn.Linear(layers[-1], out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        x = self.embedding_layer(x)
        for layer in self.layers:
            x = layer(x)
        return self.regression_layer(x)


class ConstantWidthSimpleFFNN(SimpleFeedForwardNN):
    """Plain feed-forward network with constant-width hidden layers and no residual connections.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        hidden_size: Size of all hidden layers.
        num_layers: Number of hidden layers (must be > 0).
        activation: Activation function (default: gelu).
        normalize: Type of normalization ('batch', 'layer', or None).
        dropout: Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            layers=[hidden_size] * num_layers,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )
