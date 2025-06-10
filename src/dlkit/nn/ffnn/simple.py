from collections.abc import Sequence, Callable
from typing import Literal

import torch.nn as nn
from torch import Tensor
from dlkit.nn.primitives import DenseBlock, SkipConnection


class FeedForwardNN(nn.Module):
    def __init__(
        self,
        *,
        shape: dict[str, tuple[int, ...]],
        layers: Sequence[int],
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = len(layers) - 1
        self.activation = activation
        feature_size = shape["features"][0]
        targets_size = shape["targets"][0]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(feature_size, layers[0]))

        for i in range(self.num_layers - 1):
            self.layers.append(
                SkipConnection(
                    DenseBlock(
                        layers[i],
                        layers[i + 1],
                        activation=activation,
                        normalize=normalize,
                        dropout=dropout,
                    ),
                    layer_type="linear",
                )
            )

        self.layers.append(nn.Linear(layers[-1], targets_size))

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class ConstantWidthFFNN(FeedForwardNN):
    def __init__(
        self,
        shape: dict[str, tuple[int, ...]],
        hidden_size: int | None = None,
        num_layers: int | None = None,
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        """Hidden layers with constant width and residual connections.

        Args:
                    input_size (int): Size of the input layer.
            output_size (int): Size of the output layer.
            hidden_size (int): Size of the hidden layers.
            num_layers (int): Number of hidden layers.
        """
        layers = [hidden_size] * num_layers
        super().__init__(
            shape=shape,
            layers=layers,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
            **kwargs,
        )
