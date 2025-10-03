"""Normalization strategies for metrics following Strategy Pattern."""

import torch
from torch import Tensor


class VarianceNormalizer:
    """Normalize by target variance."""

    @property
    def name(self) -> str:
        return "variance"

    @staticmethod
    def normalize(values: Tensor, reference: Tensor, eps: float = 1e-8) -> Tensor:
        """Normalize by variance of reference tensor."""
        variance = torch.var(reference)
        return values / (variance + eps)


class StandardDeviationNormalizer:
    """Normalize by target standard deviation."""

    @property
    def name(self) -> str:
        return "std"

    @staticmethod
    def normalize(values: Tensor, reference: Tensor, eps: float = 1e-8) -> Tensor:
        """Normalize by standard deviation of reference tensor."""
        std = torch.std(reference)
        return values / (std + eps)


class VectorNormNormalizer:
    """Normalize by vector norm of reference tensor."""

    def __init__(self, ord: int = 2, dim: int = -1):
        """Initialize with norm parameters.

        Args:
            ord: Order of the norm (default: 2 for L2 norm)
            dim: Dimension along which to compute norm
        """
        self.ord = ord
        self.dim = dim

    @property
    def name(self) -> str:
        return f"vector_norm_ord_{self.ord}_dim_{self.dim}"

    def normalize(self, values: Tensor, reference: Tensor, eps: float = 1e-8) -> Tensor:
        """Normalize by vector norm of reference tensor along specified dimension."""
        norm = torch.linalg.vector_norm(reference, ord=self.ord, dim=self.dim, keepdim=True)
        return values / (norm + eps)


class NaiveForecastNormalizer:
    """Normalize by naive forecast error (for time series)."""

    @property
    def name(self) -> str:
        return "naive_forecast"

    @staticmethod
    def normalize(values: Tensor, reference: Tensor, eps: float = 1e-8) -> Tensor:
        """Normalize by naive forecast error (difference between consecutive values)."""
        # Compute naive forecast error: |y[t] - y[t-1]|
        naive_error = torch.abs(reference[..., 1:] - reference[..., :-1])
        naive_scale = torch.mean(naive_error, dim=-1, keepdim=True)

        # Expand to match values shape if needed
        if naive_scale.shape != values.shape:
            # For time series, typically need to broadcast along time dimension
            naive_scale = naive_scale.expand_as(values)

        return values / (naive_scale + eps)


# Standard normalizer instances
VARIANCE_NORMALIZER = VarianceNormalizer()
STD_NORMALIZER = StandardDeviationNormalizer()
L2_NORM_NORMALIZER = VectorNormNormalizer(ord=2, dim=-1)
L1_NORM_NORMALIZER = VectorNormNormalizer(ord=1, dim=-1)
NAIVE_FORECAST_NORMALIZER = NaiveForecastNormalizer()
