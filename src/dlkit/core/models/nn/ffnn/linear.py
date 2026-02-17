from typing import Literal

import torch.nn as nn
from torch import Tensor

from dlkit.core.models.nn.base import DLKitModel


class LinearNetwork(DLKitModel):
    """A simple linear network with a single layer and optional normalization.

    This network consists of a single linear transformation with optional
    batch normalization or layer normalization.

    Args:
        in_features: Size of the input features.
        out_features: Size of the output features.
        normalize: Type of normalization to apply ('batch', 'layer', or None).
        bias: Whether to include bias in the linear layer.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        normalize: Literal["batch", "layer"] | None = None,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if normalize == "batch":
            self.norm: nn.Module | None = nn.BatchNorm1d(out_features)
        elif normalize == "layer":
            self.norm = nn.LayerNorm(out_features)
        else:
            self.norm = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the linear network.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
