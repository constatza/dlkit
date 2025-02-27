from typing import Literal, Callable

import torch
import torch.nn as nn
from torch.nn import functional as F


def agg_sum(x_in: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
    return x_in + x_out


def agg_concat(x_in: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
    return torch.cat([x_in, x_out], dim=1)  # Concatenating along the channel dimension


aggregation_functions = {
    "sum": agg_sum,
    "concat": agg_concat,
}


class ResidualBlock(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        how: Literal["sum", "concat"] = "sum",
        layer_type: Literal["conv1d", "conv2d", "linear"] = "conv1d",
        activation: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        kernel_size: int = 1,
        in_channels: int = None,
        out_channels: int = None,
    ):
        """
        Initializes the ResidualBlock.

        Args:
            module (nn.Module): The module to apply to the input.
            how (str): Aggregation method to combine input and module output.
                             Options: 'sum', 'concat', 'mean', 'max', 'min', 'weighted_sum'. Defaults to 'sum'.
            layer_type (str): Type of layer to use for aggregation. Options: 'conv1d', 'conv2d', 'linear'.
            activation
        """
        super(ResidualBlock, self).__init__()
        if hasattr(module, "in_channels") and hasattr(module, "out_channels"):
            self.in_channels = module.in_channels
            self.out_channels = module.out_channels
        elif hasattr(module, "in_features") and hasattr(module, "out_features"):
            self.in_channels = module.in_features
            self.out_channels = module.out_features
        else:
            self.in_channels = in_channels
            self.out_channels = out_channels

        if self.in_channels is None or self.out_channels is None:
            raise ValueError("in_channels and out_channels must be specified")

        self.kernel_size = kernel_size
        self.activation = activation
        self.layer_type = layer_type
        self.module = module

        self.aggregation_function = aggregation_functions[how]

        if self.in_channels == self.out_channels:
            self.reduce_layer = nn.Identity()
        elif layer_type == "conv1d":
            self.reduce_layer = nn.Conv1d(
                self.in_channels, self.out_channels, self.kernel_size
            )
        elif layer_type == "conv2d":
            self.reduce_layer = nn.Conv2d(
                self.in_channels, self.out_channels, self.kernel_size
            )
        elif layer_type == "linear":
            self.reduce_layer = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_out = self.module(x_in)
        agg_out = self.aggregation_function(self.reduce_layer(x_in), x_out)
        return self.activation(agg_out)
