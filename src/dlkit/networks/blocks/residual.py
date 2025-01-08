import torch
import torch.nn as nn

def agg_sum(x_in, x_out):
    return x_in + x_out

def agg_concat(x_in, x_out):
    return torch.cat([x_in, x_out], dim=1)  # Concatenating along the channel dimension

def agg_mean(x_in, x_out):
    return (x_in + x_out) / 2

def agg_max(x_in, x_out):
    return torch.max(x_in, x_out)

def agg_min(x_in, x_out):
    return torch.min(x_in, x_out)

class WeightedSumAggregator(nn.Module):
    def __init__(self):
        super(WeightedSumAggregator, self).__init__()
        self.w = nn.Parameter(torch.tensor(0.5))  # Learnable parameter w

    def forward(self, x_in, x_out):
        return self.w * x_in + (1 - self.w) * x_out

class ResidualBlock(nn.Module):
    def __init__(self, module: nn.Module, aggregator: str = "sum"):
        """
        Initializes the ResidualBlock.

        Args:
            module (nn.Module): The module to apply to the input.
            aggregator (str): Aggregation method to combine input and module output.
                             Options: 'sum', 'concat', 'mean', 'max', 'min', 'weighted_sum'. Defaults to 'sum'.
        """
        super(ResidualBlock, self).__init__()

        aggregators = {
            "sum": agg_sum,
            "concat": agg_concat,
            "mean": agg_mean,
            "max": agg_max,
            "min": agg_min,
            "weighted_sum": WeightedSumAggregator()
        }

        if aggregator not in aggregators:
            raise ValueError("Aggregator must be one of 'sum', 'concat', 'mean', 'max', 'min', or 'weighted_sum'")

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

