from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from dlkit.domain.nn.types import NormalizerName
from dlkit.domain.nn.utils import make_norm_layer


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
    ):
        """Initializes a DenseBlock.

        Parameters:
            in_features (int): Number of input features to the layer.
            out_features (int): Number of output features from the layer.
            activation (Callable[[torch.Tensor], torch.Tensor], optional): Activation function to be used in the layer. Defaults to F.gelu.
            normalize (str | None, optional): Normalization type ('layer', 'batch', or None). Defaults to None.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.norm = make_norm_layer(normalize, in_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.fc1 = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return x
