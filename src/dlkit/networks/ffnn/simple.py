from collections.abc import Sequence, Callable

import torch.nn as nn
from torch import Tensor


class FeedForwardNN(nn.Module):
    def __init__(
        self,
        shape: dict[str, tuple[int, ...]],
        layers: Sequence[int],
        activation: Callable[[Tensor], Tensor] = nn.functional.relu,
        layer_norm: bool = False,
        dropout: float = 0,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.num_layers = len(layers) - 1
        self.activation = activation
        feature_size = shape["features"][0]
        targets_size = shape["targets"][0]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(feature_size, layers[0]))

        for i in range(1, self.num_layers - 1):
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            if layer_norm:
                self.layers.append(nn.LayerNorm(layers[i + 1]))
            elif batch_norm:
                self.layers.append(nn.BatchNorm1d(layers[i + 1]))
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        self.layers.append(nn.Linear(layers[-1], targets_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

    def predict_step(self, batch, batch_idx):
        """Predict step for the model."""
        x = batch[0]
        predictions = self.forward(x)
        return predictions


class ConstantWidthFFNN(FeedForwardNN):
    def __init__(
        self,
        shape: dict[str, tuple[int, ...]],
        hidden_size: int | None = None,
        num_layers: int | None = None,
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
        super().__init__(shape, layers, **kwargs)

    def forward(self, x):
        x = self.layers[0](x)  # Initial layer
        for layer in self.layers[1:-1]:  # Skip the last layer
            residual = self.activation(layer(x))
            x = x + residual
        x = self.layers[-1](x)
        return x
