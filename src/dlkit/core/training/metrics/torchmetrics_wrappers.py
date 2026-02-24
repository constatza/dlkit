"""TorchMetrics wrappers for custom metrics.

This module provides torchmetrics.Metric wrappers for our custom functional metrics,
enabling:
    - MetricCollection compatibility
    - MLflow logging integration
    - Distributed training support (DDP)
    - State accumulation across batches

All wrappers delegate computation to functional implementations in functional.py,
following the separation of concerns: wrappers handle state, functionals handle logic.
"""

import torch
from torch import Tensor
from torchmetrics import Metric

from .functional import (
    _normalized_vector_norm_update,
    _normalized_vector_norm_compute,
    _absolute_vector_norm_update,
    _absolute_vector_norm_compute,
    _energy_norm_update,
    _energy_norm_compute,
    _relative_energy_norm_update,
    _relative_energy_norm_compute,
    _temporal_derivative_update,
    _temporal_derivative_compute,
)


class NormalizedVectorNormError(Metric):
    """TorchMetrics wrapper for normalized vector norm error metric.

    Computes: mean(||pred - target|| / ||target||) across batches.

    This metric normalizes error by target magnitude, providing relative
    error measurement. Accumulates state across batches for proper
    distributed training support.

    Shape Contract:
        Input: (B, ..., D) where D is vector_dim
            B = Batch size
            D = Vector dimension (along which to compute norms)
            ... = Arbitrary intermediate dimensions

        Output: Scalar tensor after compute()

    Attributes:
        sum_errors: Accumulated sum of normalized errors
        total: Total number of samples processed

    Args:
        vector_dim: Dimension along which vectors are defined (default: -1)
        norm_ord: Order of norm (1=L1, 2=L2, float('inf')=Linf) (default: 2)
        eps: Numerical stability epsilon for division (default: 1e-8)
        **kwargs: Additional arguments passed to torchmetrics.Metric

    Examples:
        >>> from torchmetrics import MetricCollection
        >>> import torch
        >>>
        >>> # Single metric usage
        >>> metric = NormalizedVectorNormError(vector_dim=-1, norm_ord=2)
        >>> preds = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        >>> target = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
        >>>
        >>> metric.update(preds, target)
        >>> error = metric.compute()
        >>> print(f"Error: {error:.4f}")
        >>>
        >>> # Use in MetricCollection
        >>> metrics = MetricCollection([
        ...     NormalizedVectorNormError(norm_ord=1),
        ...     NormalizedVectorNormError(norm_ord=2),
        ... ])
        >>> metrics.update(preds, target)
        >>> results = metrics.compute()
    """

    def __init__(
        self,
        vector_dim: int = -1,
        norm_ord: int = 2,
        eps: float = 1e-8,
        **kwargs,
    ):
        """Initialize normalized vector norm error metric."""
        super().__init__(**kwargs)

        # Store configuration
        self.vector_dim = vector_dim
        self.norm_ord = norm_ord
        self.eps = eps

        # Add metric states (will be synchronized across processes)
        self.add_state(
            "sum_errors",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "total",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state with batch of predictions and targets.

        Args:
            preds: Predicted vectors with shape (B, ..., D)
            target: Ground truth vectors with shape (B, ..., D)

        Raises:
            ValueError: If inputs are not at least 2D
            ValueError: If shapes don't match
        """
        # Compute per-sample normalized errors (not aggregated)
        errors = _normalized_vector_norm_update(
            preds, target, self.norm_ord, self.vector_dim, self.eps
        )

        # Accumulate into state
        self.sum_errors += errors.sum()
        self.total += errors.numel()

    def compute(self) -> Tensor:
        """Compute final metric value from accumulated state.

        Returns:
            Scalar tensor with mean normalized error across all batches
        """
        return _normalized_vector_norm_compute(self.sum_errors, self.total)


class TemporalDerivativeError(Metric):
    """TorchMetrics wrapper for temporal derivative error metric.

    Computes: mean squared error of nth temporal derivatives.

    Measures how well predictions match target dynamics (velocity, acceleration, etc.)
    rather than just static values. Useful for physics-based sequential predictions.

    Shape Contract:
        Input: (B, T, D) - 3D tensor required
            B = Batch size
            T = Temporal/sequence dimension (must be >= n+1)
            D = Feature dimension at each time step

        Output: Scalar tensor after compute()

    Temporal Requirements:
        - Input must be exactly 3D
        - T must be >= n+1 (need n+1 points for nth derivative)
        - Output temporal dimension after derivative: T-n

    Attributes:
        sum_squared_errors: Accumulated sum of squared derivative errors
        total: Total number of error values processed

    Args:
        n: Derivative order (1=velocity, 2=acceleration, etc.) (default: 1)
        derivative_dim: Index of temporal dimension (default: 1 for middle dim)
        **kwargs: Additional arguments passed to torchmetrics.Metric

    Examples:
        >>> from torchmetrics import MetricCollection
        >>> import torch
        >>>
        >>> # Velocity error (1st derivative)
        >>> metric = TemporalDerivativeError(n=1, derivative_dim=1)
        >>> preds = torch.randn(4, 10, 3)  # (batch=4, time=10, features=3)
        >>> target = torch.randn(4, 10, 3)
        >>>
        >>> metric.update(preds, target)
        >>> vel_error = metric.compute()
        >>>
        >>> # Acceleration error (2nd derivative)
        >>> accel_metric = TemporalDerivativeError(n=2, derivative_dim=1)
        >>> accel_metric.update(preds, target)
        >>> accel_error = accel_metric.compute()
        >>>
        >>> # Use in MetricCollection
        >>> metrics = MetricCollection({
        ...     'velocity_error': TemporalDerivativeError(n=1),
        ...     'acceleration_error': TemporalDerivativeError(n=2),
        ... })
    """

    def __init__(
        self,
        n: int = 1,
        derivative_dim: int = 1,
        **kwargs,
    ):
        """Initialize temporal derivative error metric."""
        super().__init__(**kwargs)

        # Store configuration
        self.n = n
        self.derivative_dim = derivative_dim

        # Add metric states
        self.add_state(
            "sum_squared_errors",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "total",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state with batch of temporal sequences.

        Args:
            preds: Predicted sequence with shape (B, T, D)
            target: Ground truth sequence with shape (B, T, D)

        Raises:
            ValueError: If inputs are not 3D
            ValueError: If T < n+1
            ValueError: If shapes don't match
        """
        # Compute per-element squared derivative errors
        squared_errors = _temporal_derivative_update(
            preds, target, self.n, self.derivative_dim
        )

        # Accumulate into state
        self.sum_squared_errors += squared_errors.sum()
        self.total += squared_errors.numel()

    def compute(self) -> Tensor:
        """Compute final metric value from accumulated state.

        Returns:
            Scalar tensor with mean squared derivative error across all batches
        """
        return _temporal_derivative_compute(self.sum_squared_errors, self.total)


class AbsoluteVectorNormError(Metric):
    """TorchMetrics wrapper for absolute vector norm error metric.

    Computes: mean(||pred - target||_ord) across batches.

    Absolute counterpart to NormalizedVectorNormError (relative). Accumulates
    state across batches for distributed training support.

    Shape Contract:
        Input: (B, ..., D) where D is vector_dim

    Attributes:
        sum_norms: Accumulated sum of per-sample norm errors
        total: Total number of samples processed

    Args:
        vector_dim: Dimension along which vectors are defined (default: -1)
        norm_ord: Order of norm (1=L1, 2=L2, float('inf')=Linf) (default: 2)
        **kwargs: Additional arguments passed to torchmetrics.Metric
    """

    def __init__(self, vector_dim: int = -1, norm_ord: int = 2, **kwargs):
        """Initialize absolute vector norm error metric."""
        super().__init__(**kwargs)
        self.vector_dim = vector_dim
        self.norm_ord = norm_ord
        self.add_state("sum_norms", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state with batch of predictions and targets.

        Args:
            preds: Predicted vectors with shape (B, ..., D)
            target: Ground truth vectors with shape (B, ..., D)
        """
        per_sample = _absolute_vector_norm_update(
            preds, target, self.norm_ord, self.vector_dim
        )
        self.sum_norms += per_sample.sum()
        self.total += per_sample.numel()

    def compute(self) -> Tensor:
        """Compute final metric value from accumulated state.

        Returns:
            Scalar tensor with mean absolute vector norm error across all batches
        """
        return _absolute_vector_norm_compute(self.sum_norms, self.total)


class EnergyNormError(Metric):
    """TorchMetrics wrapper for absolute energy norm (A-norm) error.

    Computes: mean(||pred - target||_A) across batches, where
    ||u||_A = sqrt(u^T A u) for a positive (semi-)definite matrix A.

    When A = I this reduces to AbsoluteVectorNormError with ord=2.

    Shape Contract:
        preds/target: (B, D)
        matrix: (B, D, D) per-sample, or (D, D) shared

    Attributes:
        sum_norms: Accumulated sum of per-sample A-norm errors
        total: Total number of samples processed

    Args:
        **kwargs: Additional arguments passed to torchmetrics.Metric
    """

    def __init__(self, **kwargs):
        """Initialize energy norm error metric."""
        super().__init__(**kwargs)
        self.add_state("sum_norms", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, matrix: Tensor) -> None:
        """Update metric state with batch of predictions, targets and matrix.

        Args:
            preds: Predicted vectors with shape (B, D)
            target: Ground truth vectors with shape (B, D)
            matrix: Positive (semi-)definite matrix with shape (B, D, D) or (D, D)
        """
        per_sample = _energy_norm_update(preds, target, matrix)
        self.sum_norms += per_sample.sum()
        self.total += per_sample.numel()

    def compute(self) -> Tensor:
        """Compute final metric value from accumulated state.

        Returns:
            Scalar tensor with mean absolute energy norm error across all batches
        """
        return _energy_norm_compute(self.sum_norms, self.total)


class RelativeEnergyNormError(Metric):
    """TorchMetrics wrapper for relative energy norm (A-norm) error.

    Computes: mean(||pred - target||_A / ||target||_A) across batches.

    Provides a dimensionless relative error in the energy norm metric.
    Analogous to the loss used in preconditioner learning (Notay loss)
    without PCG-specific semantics.

    Shape Contract:
        preds/target: (B, D)
        matrix: (B, D, D) per-sample, or (D, D) shared

    Attributes:
        sum_norms: Accumulated sum of per-sample relative energy norm errors
        total: Total number of samples processed

    Args:
        eps: Numerical stability epsilon for division (default: 1e-8)
        **kwargs: Additional arguments passed to torchmetrics.Metric
    """

    def __init__(self, eps: float = 1e-8, **kwargs):
        """Initialize relative energy norm error metric."""
        super().__init__(**kwargs)
        self.eps = eps
        self.add_state("sum_norms", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, matrix: Tensor) -> None:
        """Update metric state with batch of predictions, targets and matrix.

        Args:
            preds: Predicted vectors with shape (B, D)
            target: Ground truth vectors with shape (B, D)
            matrix: Positive (semi-)definite matrix with shape (B, D, D) or (D, D)
        """
        per_sample = _relative_energy_norm_update(preds, target, matrix, eps=self.eps)
        self.sum_norms += per_sample.sum()
        self.total += per_sample.numel()

    def compute(self) -> Tensor:
        """Compute final metric value from accumulated state.

        Returns:
            Scalar tensor with mean relative energy norm error across all batches
        """
        return _relative_energy_norm_compute(self.sum_norms, self.total)


__all__ = [
    "NormalizedVectorNormError",
    "TemporalDerivativeError",
    "AbsoluteVectorNormError",
    "EnergyNormError",
    "RelativeEnergyNormError",
]
