"""Concrete metric implementations using the new architecture."""

import torch
from torch import Tensor

from .base import BaseMetric
from .protocols import IAggregator, INormalizer
from .aggregators import MEAN_AGGREGATOR, L2_AGGREGATOR
from .normalizers import VARIANCE_NORMALIZER


class MeanSquaredErrorMetric(BaseMetric):
    """Mean Squared Error metric."""

    def __init__(
        self, aggregator: IAggregator | None = None, normalizer: INormalizer | None = None, **kwargs
    ):
        super().__init__(
            name="mse", aggregator=aggregator or MEAN_AGGREGATOR, normalizer=normalizer, **kwargs
        )

    def _compute_raw_error(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute squared differences."""
        return torch.pow(predictions - targets, 2)


class MeanAbsoluteErrorMetric(BaseMetric):
    """Mean Absolute Error metric."""

    def __init__(
        self, aggregator: IAggregator | None = None, normalizer: INormalizer | None = None, **kwargs
    ):
        super().__init__(
            name="mae", aggregator=aggregator or MEAN_AGGREGATOR, normalizer=normalizer, **kwargs
        )

    def _compute_raw_error(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute absolute differences."""
        return torch.abs(predictions - targets)


class RootMeanSquaredErrorMetric(BaseMetric):
    """Root Mean Squared Error metric."""

    def __init__(
        self, aggregator: IAggregator | None = None, normalizer: INormalizer | None = None, **kwargs
    ):
        super().__init__(
            name="rmse", aggregator=aggregator or L2_AGGREGATOR, normalizer=normalizer, **kwargs
        )

    def _compute_raw_error(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute differences for RMS calculation."""
        return predictions - targets

    def _should_normalize(self) -> bool:
        """RMSE typically doesn't normalize before taking the norm."""
        return False

    def _post_process(self, result: Tensor) -> Tensor:
        """Apply square root to the aggregated result."""
        # The L2_AGGREGATOR already computes the norm, which includes sqrt
        return result


class NormalizedVectorNormErrorMetric(BaseMetric):
    """Normalized Vector Norm Error for 2D

    Computes the error between predicted and target vectors, where each vector
    is normalized by the target vector's norm. This is particularly useful for
    2D dataflow where each row represents a vector and you want to measure relative
    error normalized by the target magnitude.

    For 2D input tensors of shape (batch, features):
    - Each row is treated as a vector
    - Error vectors are computed as (predictions - targets)
    - Each error vector is normalized by the corresponding target vector norm
    - Final aggregation can be customized via aggregator parameter
    """

    def __init__(
        self,
        vector_dim: int = -1,
        norm_ord: int = 2,
        aggregator: IAggregator | None = None,
        normalizer: INormalizer | None = None,  # Accept normalizer for compatibility
        eps: float = 1e-8,
        **kwargs,
    ):
        """Initialize normalized vector norm error metric.

        Args:
            vector_dim: Dimension along which vectors are defined (default: -1, last dim)
            norm_ord: Order of the norm (default: 2 for L2 norm)
            aggregator: Strategy for final aggregation (default: mean)
            normalizer: Ignored - normalization is handled manually (for compatibility)
            eps: Small value for numerical stability
            **kwargs: Additional parameters
        """
        # Accept normalizer parameter but don't use it (for factory compatibility)
        super().__init__(
            name=f"normalized_vector_norm_error_ord_{norm_ord}",
            aggregator=aggregator or MEAN_AGGREGATOR,
            normalizer=None,  # We handle normalization manually
            vector_dim=vector_dim,
            norm_ord=norm_ord,
            eps=eps,
            **kwargs,
        )

    def _compute_raw_error(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute normalized vector norm errors.

        For each vector (row), computes:
        ||prediction_vector - target_vector|| / ||target_vector||

        Args:
            predictions: Predicted vectors of shape (batch, features)
            targets: Target vectors of shape (batch, features)

        Returns:
            Normalized errors of shape (batch,) where each element is the
            normalized error for the corresponding vector
        """
        vector_dim = self._params["vector_dim"]
        norm_ord = self._params["norm_ord"]
        eps = self._params["eps"]

        # Compute error vectors
        error_vectors = predictions - targets

        # Compute norms of error vectors and target vectors
        error_norms = torch.linalg.vector_norm(error_vectors, ord=norm_ord, dim=vector_dim)
        target_norms = torch.linalg.vector_norm(targets, ord=norm_ord, dim=vector_dim)

        # Normalize error norms by target norms (with epsilon for stability)
        normalized_errors = error_norms / (target_norms + eps)

        return normalized_errors

    def _should_normalize(self) -> bool:
        """Normalization is handled in _compute_raw_error."""
        return False

    def _validate_inputs(self, predictions: Tensor, targets: Tensor) -> None:
        """Validate inputs for vector norm computation."""
        super()._validate_inputs(predictions, targets)

        # Ensure we have at least 2D tensors for vector operations
        if predictions.dim() < 2:
            raise ValueError(
                f"Expected at least 2D tensors for vector operations, "
                f"got {predictions.dim()}D tensors"
            )

        vector_dim = self._params["vector_dim"]
        if abs(vector_dim) > predictions.dim():
            raise ValueError(
                f"Vector dimension {vector_dim} is out of bounds for {predictions.dim()}D tensor"
            )


class MSEOverVarianceMetric(BaseMetric):
    """MSE over variance metric (legacy compatibility)."""

    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(
            name="mse_over_var",
            aggregator=MEAN_AGGREGATOR,
            normalizer=VARIANCE_NORMALIZER,
            eps=eps,
            **kwargs,
        )

    def _compute_raw_error(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute squared differences."""
        return torch.pow(predictions - targets, 2)


class TemporalDerivativeMetric(BaseMetric):
    """Metric for temporal derivative errors."""

    def __init__(
        self,
        n: int = 1,
        derivative_dim: int = -1,
        aggregator: IAggregator | None = None,
        normalizer: INormalizer | None = None,
        **kwargs,
    ):
        """Initialize temporal derivative metric.

        Args:
            n: Order of derivative (default: 1 for first derivative)
            derivative_dim: Dimension along which to compute derivative
            aggregator: Aggregation strategy
            normalizer: Normalization strategy
            **kwargs: Additional parameters
        """
        super().__init__(
            name=f"temporal_derivative_order_{n}",
            aggregator=aggregator or MEAN_AGGREGATOR,
            normalizer=normalizer,
            derivative_order=n,
            derivative_dim=derivative_dim,
            **kwargs,
        )

    def _compute_raw_error(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute derivative of error."""
        n = self._params["derivative_order"]
        dim = self._params["derivative_dim"]

        # Compute error first, then take derivative
        error = predictions - targets
        derivative_error = torch.diff(error, n=n, dim=dim)

        return torch.pow(derivative_error, 2)  # Squared derivative error
