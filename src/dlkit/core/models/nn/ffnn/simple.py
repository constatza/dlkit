from collections.abc import Callable, Sequence
from typing import Literal

from torch import Tensor, nn

from dlkit.core.models.nn.base import DLKitModel
from dlkit.core.models.nn.primitives import DenseBlock, SkipConnection


class FeedForwardNN(DLKitModel):
    """Feed-forward neural network with skip connections and configurable layers.

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
    ):
        super().__init__()
        self.num_layers = len(layers)
        self.activation = activation

        self.layers = nn.ModuleList()
        self.embedding_layer = nn.Linear(in_features, layers[0])

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


class ConstantWidthFFNN(FeedForwardNN):
    """Feed-forward network with constant-width hidden layers and residual connections.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        hidden_size: Size of all hidden layers.
        num_layers: Number of hidden layers.
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
    ):
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
