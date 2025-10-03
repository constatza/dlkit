from typing import Literal

import torch.nn as nn
from torch import Tensor

from dlkit.core.models.nn.base import ShapeAwareModel
from dlkit.core.shape_specs import IShapeSpec


class LinearNetwork(ShapeAwareModel):
    """A simple linear network with a single layer and optional normalization.

    This network consists of a single linear transformation with optional
    batch normalization or layer normalization.

    This model is ideal for testing scenarios where a simple, minimal model
    is needed without the complexity of deeper architectures.

    Args:
        shape_spec: Shape specification containing input and output shape information
        normalize: Type of normalization to apply ('batch', 'layer', or None)
        bias: Whether to include bias in the linear layer
        shape: Dictionary containing input and output shapes (deprecated, use shape_spec)
        input_size: Size of the input features (deprecated, use shape_spec)
        output_size: Size of the output features (deprecated, use shape_spec)
    """

    def __init__(
        self,
        *,
        unified_shape: IShapeSpec,
        normalize: Literal["batch", "layer"] | None = None,
        bias: bool = True,
        **kwargs,
    ):
        # Call ShapeAwareModel constructor
        super().__init__(unified_shape=unified_shape, **kwargs)

        # Build layers based on shape
        self._build_layers(normalize, bias)

    def _build_layers(self, normalize: Literal["batch", "layer"] | None, bias: bool) -> None:
        """Build the network layers based on shape configuration."""
        shape_spec = self.get_unified_shape()

        input_shape = shape_spec.get_input_shape()
        output_shape = shape_spec.get_output_shape()

        if input_shape is None or output_shape is None:
            raise ValueError("LinearNetwork requires both input and output shape information")

        input_size = input_shape[0]
        output_size = output_shape[0]

        self.linear = nn.Linear(input_size, output_size, bias=bias)

        # Add normalization if specified
        if normalize == "batch":
            self.norm = nn.BatchNorm1d(output_size)
        elif normalize == "layer":
            self.norm = nn.LayerNorm(output_size)
        else:
            self.norm = None

        # Apply precision to newly created parameters
        self.ensure_precision_applied()

    def accepts_shape(self, shape_spec: IShapeSpec) -> bool:
        """Check if this LinearNetwork can accept the given shape specification."""
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

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the linear network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        x = self.linear(x)

        if self.norm is not None:
            x = self.norm(x)

        return x
