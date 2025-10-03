"""Aggregation strategies for metrics following Strategy Pattern."""

import torch
from torch import Tensor


class MeanAggregator:
    """Mean aggregation strategy."""

    @property
    def name(self) -> str:
        return "mean"

    @staticmethod
    def aggregate(values: Tensor, dim: int | None = None) -> Tensor:
        """Compute mean along specified dimension."""
        return torch.mean(values, dim=dim)


class SumAggregator:
    """Sum aggregation strategy."""

    @property
    def name(self) -> str:
        return "sum"

    @staticmethod
    def aggregate(values: Tensor, dim: int | None = None) -> Tensor:
        """Compute sum along specified dimension."""
        return torch.sum(values, dim=dim)


class VectorNormAggregator:
    """Vector norm aggregation strategy."""

    def __init__(self, ord: int = 2):
        """Initialize with norm order.

        Args:
            ord: Order of the norm (default: 2 for L2 norm)
        """
        self.ord = ord

    @property
    def name(self) -> str:
        return f"vector_norm_ord_{self.ord}"

    def aggregate(self, values: Tensor, dim: int | None = None) -> Tensor:
        """Compute vector norm along specified dimension."""
        if dim is None:
            # Flatten all dimensions for global norm
            values = values.flatten()
            dim = 0
        return torch.linalg.vector_norm(values, ord=self.ord, dim=dim)


class StdAggregator:
    """Standard deviation aggregation strategy."""

    @property
    def name(self) -> str:
        return "std"

    @staticmethod
    def aggregate(values: Tensor, dim: int | None = None) -> Tensor:
        """Compute standard deviation along specified dimension."""
        return torch.std(values, dim=dim)


# Standard aggregator instances
MEAN_AGGREGATOR = MeanAggregator()
SUM_AGGREGATOR = SumAggregator()
L2_AGGREGATOR = VectorNormAggregator(ord=2)
L1_AGGREGATOR = VectorNormAggregator(ord=1)
STD_AGGREGATOR = StdAggregator()
