from typing import Literal

import torch
import torch.nn as nn
from torch.nn import functional as F


class Aggregator1d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 1,
        aggregator: Literal["sum", "concat"] = "sum",
    ):
        super(Aggregator1d, self).__init__()
        if in_channels != out_channels:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
        else:
            self.conv1 = nn.Identity()

        if aggregator == "sum":
            self.aggregator = agg_sum
        elif aggregator == "concat":
            self.aggregator = agg_concat

    def forward(self, x_in, x_out):
        return self.aggregator(self.conv1(x_in), x_out)


def agg_sum(x_in, x_out):
    return x_in + x_out


def agg_concat(x_in, x_out):
    return torch.cat([x_in, x_out], dim=1)  # Concatenating along the channel dimension


class ResidualBlock(nn.Module):
    def __init__(self, module: nn.Module, aggregator: str = "sum", activation=F.gelu):
        """
        Initializes the ResidualBlock.

        Args:
            module (nn.Module): The module to apply to the input.
            aggregator (str): Aggregation method to combine input and module output.
                             Options: 'sum', 'concat', 'mean', 'max', 'min', 'weighted_sum'. Defaults to 'sum'.
            activation (nn.Module): The activation function to apply to the output.
        """
        super(ResidualBlock, self).__init__()
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.activation = activation

        aggregators = nn.ModuleDict(
            {
                "sum": Aggregator1d(
                    self.in_channels, self.out_channels, aggregator="sum"
                ),
                "concat": Aggregator1d(
                    self.in_channels, self.out_channels, aggregator="concat"
                ),
            }
        )

        if aggregator not in aggregators:
            raise ValueError(
                f"Aggregator must be one of f{list(aggregators.keys())}, but got {aggregator}"
            )

        self.module = module
        self.aggregator = aggregators[aggregator]

    def forward(self, x):
        """
        Forward pass of the ResidualBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The result of the aggregation.
        """
        output = self.module(x)
        return self.aggregator(x, output)
