from typing import Literal
from collections.abc import Callable

import torch
import torch.nn as nn


def agg_sum(x_in: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
    return x_in + x_out


def agg_concat(x_in: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
    return torch.cat([x_in, x_out], dim=1)  # Concatenating along the channel dimension


aggregation_functions = {
    "sum": agg_sum,
    "concat": agg_concat,
}


class SkipConnection(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        how: Literal["sum", "concat"] = "sum",
        layer_type: Literal["conv1d", "conv2d", "linear"] = "conv1d",
        activation: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        in_channels: int = None,
        out_channels: int = None,
        kernel_size: int = 1,
        stride: int = 1,
        bias: bool = True,
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
        super(SkipConnection, self).__init__()
        if hasattr(module, "in_channels") and hasattr(module, "out_channels"):
            self.in_channels = module.in_channels
            self.out_channels = module.out_channels
        elif hasattr(module, "in_features") and hasattr(module, "out_features"):
            self.in_channels = module.in_features
            self.out_channels = module.out_features
        else:
            self.in_channels = in_channels
            self.out_channels = out_channels

        if hasattr(module, "dilation"):
            self.dilation = module.dilation
        if hasattr(module, "stride"):
            self.stride = module.stride

        if self.in_channels is None or self.out_channels is None:
            raise ValueError("in_channels and out_channels must be specified")

        self.reduce_layer = select_skip_layers(
            layer_type,
            self.in_channels,
            self.out_channels,
            kernel_size,
            stride,
            bias=bias,
        )
        self.kernel_size = kernel_size
        self.activation = activation
        self.layer_type = layer_type
        self.module = module

        self.aggregation_function = aggregation_functions[how]

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_out = self.module(x_in)
        skip = self.reduce_layer(x_in)
        # if skip.shape[2:] != x_out.shape[2:]:
        #     skip = torch.nn.functional.interpolate(skip, size=x_out.shape[2:])
        agg_out = self.aggregation_function(skip, x_out)
        return self.activation(agg_out)


def select_skip_layers(
    layer_type, in_channels, out_channels, kernel_size, stride, bias=True
):
    if in_channels == out_channels:
        return nn.Identity()
    if layer_type == "conv1d":
        return nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=bias)
    if layer_type == "conv2d":
        return nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=bias)
    if layer_type == "linear":
        return nn.Linear(in_channels, out_channels, bias=False)
