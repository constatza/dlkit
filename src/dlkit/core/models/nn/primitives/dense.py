from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        normalize: Literal["layer", "batch"] | None = None,
        dropout: float = 0.0,
    ):
        """Initializes a DenseBlock.

        Parameters:
            in_features (int): Number of input x to the layer.
            out_features (int): Number of output x from the layer.
            activation (Callable[[torch.Tensor], torch.Tensor], optional): Activation function to be used in the layer. Defaults to F.gelu.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.norm = nn.Identity()
        if normalize == "layer":
            self.norm = nn.LayerNorm(in_features)
        elif normalize == "batch":
            self.norm = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.fc1 = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return x
