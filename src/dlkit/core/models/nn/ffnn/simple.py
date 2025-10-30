from collections.abc import Sequence, Callable
from typing import Literal

import torch.nn as nn
from torch import Tensor
from dlkit.core.models.nn.primitives import DenseBlock, SkipConnection
from dlkit.core.shape_specs import IShapeSpec
from dlkit.core.models.nn.base import ShapeAwareModel


class FeedForwardNN(ShapeAwareModel):
    def __init__(
        self,
        *,
        unified_shape: IShapeSpec,
        layers: Sequence[int],
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        # Call ShapeAwareModel constructor with unified_shape
        super().__init__(unified_shape=unified_shape, **kwargs)

        # Build layers based on shape
        self._build_layers(layers, activation, normalize, dropout)

    def _build_layers(
        self,
        layers: Sequence[int],
        activation: Callable[[Tensor], Tensor],
        normalize: Literal["batch", "layer"] | None,
        dropout: float
    ) -> None:
        """Build the network layers based on shape configuration."""
        shape_spec = self.get_unified_shape()

        input_shape = shape_spec.get_input_shape()
        output_shape = shape_spec.get_output_shape()

        if input_shape is None:
            raise ValueError("FeedForwardNN requires input shape information")
        if output_shape is None:
            raise ValueError("FeedForwardNN requires output shape information")

        self.num_layers = len(layers) - 1
        self.activation = activation

        feature_size = input_shape[0]
        targets_size = output_shape[0]
        self.layers = nn.ModuleList()
        self.embedding_layer = nn.Linear(feature_size, layers[0])

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

        self.regression_layer = nn.Linear(layers[-1], targets_size)

    def accepts_shape(self, shape_spec: IShapeSpec) -> bool:
        """Check if this FeedForwardNN can accept the given shape specification."""
        # Explicitly reject NullShapeSpec - FeedForwardNN needs real shapes
        from dlkit.core.shape_specs import NullShapeSpec
        if isinstance(shape_spec, NullShapeSpec):
            return False

        # Additional validation: check for required dimensions
        input_shape = shape_spec.get_input_shape()
        output_shape = shape_spec.get_output_shape()

        if input_shape is None or output_shape is None:
            return False

        # Validate that shapes are 1D (feature vectors)
        if len(input_shape) != 1 or len(output_shape) != 1:
            return False

        # Validate positive dimensions
        if input_shape[0] <= 0 or output_shape[0] <= 0:
            return False

        return True

    def forward(self, x):
        x = self.embedding_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.regression_layer(x)
        return x


class ConstantWidthFFNN(FeedForwardNN):
    def __init__(
        self,
        *,
        unified_shape: IShapeSpec,
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
        if hidden_size is None or num_layers is None:
            msg = "ConstantWidthFFNN requires explicit hidden_size and num_layers"
            raise ValueError(msg)
        if num_layers <= 0:
            msg = "num_layers must be a positive integer"
            raise ValueError(msg)

        hidden_width = hidden_size
        layer_count = num_layers
        layers = [hidden_width] * layer_count
        super().__init__(
            unified_shape=unified_shape,
            layers=layers,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
            **kwargs,
        )
