"""Shared functional implementations for losses and metrics.

This module provides pure, differentiable functions that can be used as:
    - Loss functions (for training with backpropagation)
    - Metric functions (for evaluation and logging)

All functions in this module are:
    - Pure (no side effects)
    - Differentiable (gradients flow through for losses)
    - Efficient (single-pass computation)
    - Type-safe (with proper type hints)

Design Philosophy:
    - Don't duplicate implementations from torchmetrics.functional
    - Re-export standard functions for consistency
    - Add custom differentiable functions as needed
    - Keep signatures simple: (pred, target) -> Tensor

Shape Contracts:
    All functions expect matching pred/target shapes and return scalar tensors.

Examples:
    >>> import torch
    >>> from dlkit.core.training.functional import mse, mae
    >>>
    >>> # Use as loss function
    >>> pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> target = torch.tensor([1.1, 1.9, 3.1])
    >>> loss = mse(pred, target)
    >>> loss.backward()  # Differentiable!
    >>>
    >>> # Use as metric
    >>> metric_value = mae(pred.detach(), target)
"""

from typing import Literal

import torch
from torch import Tensor

from .metrics.functional import (
    AggregatorFn,
    apply_aggregation,
    compute_error_vectors,
    compute_vector_norm,
    compute_energy_norm,
    safe_divide,
)

# ============================================================================
# STANDARD REGRESSION LOSSES/METRICS (from torchmetrics.functional)
# ============================================================================

from torchmetrics.functional.regression import (
    mean_squared_error as mse,
    mean_absolute_error as mae,
    mean_squared_log_error as msle,
    mean_absolute_percentage_error as mape,
)

# ============================================================================
# CUSTOM DIFFERENTIABLE FUNCTIONS
# ============================================================================


def huber_loss(
    preds: Tensor,
    target: Tensor,
    delta: float = 1.0,
    aggregator: AggregatorFn = torch.mean,
) -> Tensor:
    """Huber loss - smooth combination of L1 and L2 loss.

    Less sensitive to outliers than MSE. Uses L2 loss for small errors
    and L1 loss for large errors.

    Shape:
        preds:  (*) - any shape
        target: (*) - must match preds
        output: (,) - scalar (default aggregator)

    Args:
        preds: Predicted values
        target: Ground truth values
        delta: Threshold for switching from L2 to L1 (default: 1.0)
        aggregator: Reduction function applied to per-element losses (default: torch.mean)

    Returns:
        Aggregated Huber loss (differentiable)

    References:
        Huber, P. J. (1964). Robust Estimation of a Location Parameter.
    """
    error = preds - target
    abs_error = torch.abs(error)

    # Quadratic for small errors, linear for large errors
    quadratic = 0.5 * error**2
    linear = delta * (abs_error - 0.5 * delta)

    return apply_aggregation(torch.where(abs_error <= delta, quadratic, linear), aggregator)


def smooth_l1_loss(
    preds: Tensor,
    target: Tensor,
    beta: float = 1.0,
) -> Tensor:
    """Smooth L1 loss (used in object detection).

    Alias for Huber loss with different parameterization.

    Shape:
        preds:  (*) - any shape
        target: (*) - must match preds
        output: (,) - scalar

    Args:
        preds: Predicted values
        target: Ground truth values
        beta: Transition point (default: 1.0)

    Returns:
        Scalar smooth L1 loss (differentiable)
    """
    return huber_loss(preds, target, delta=beta)


def log_cosh_loss(
    preds: Tensor,
    target: Tensor,
    aggregator: AggregatorFn = torch.mean,
) -> Tensor:
    """Log-cosh loss - smooth approximation of MAE.

    Less sensitive to outliers than MSE, smoother than MAE.
    Approximately equal to (x^2 / 2) for small x, |x| for large x.

    Shape:
        preds:  (*) - any shape
        target: (*) - must match preds
        output: (,) - scalar (default aggregator)

    Args:
        preds: Predicted values
        target: Ground truth values
        aggregator: Reduction function applied to per-element losses (default: torch.mean)

    Returns:
        Aggregated log-cosh loss (differentiable)

    References:
        https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
    """
    error = preds - target
    return apply_aggregation(torch.log(torch.cosh(error)), aggregator)


def quantile_loss(
    preds: Tensor,
    target: Tensor,
    quantile: float = 0.5,
    aggregator: AggregatorFn = torch.mean,
) -> Tensor:
    """Quantile loss (pinball loss) for quantile regression.

    When quantile=0.5, equivalent to MAE. Useful for predicting
    different quantiles of the target distribution.

    Shape:
        preds:  (*) - any shape
        target: (*) - must match preds
        output: (,) - scalar (default aggregator)

    Args:
        preds: Predicted values
        target: Ground truth values
        quantile: Desired quantile in [0, 1] (default: 0.5 = median)
        aggregator: Reduction function applied to per-element losses (default: torch.mean)

    Returns:
        Aggregated quantile loss (differentiable)

    Raises:
        ValueError: If quantile not in [0, 1]
    """
    if not 0 <= quantile <= 1:
        raise ValueError(f"Quantile must be in [0, 1], got {quantile}")

    error = target - preds
    # Asymmetric penalty: heavier for underestimation (pred < target) when quantile > 0.5
    return apply_aggregation(
        torch.where(error >= 0, quantile * error, (quantile - 1) * error),
        aggregator,
    )


# ============================================================================
# NORMALIZED LOSSES
# ============================================================================


def vector_norm_loss(
    preds: Tensor,
    target: Tensor,
    ord: int = 2,
    dim: int = -1,
    aggregator: AggregatorFn = torch.mean,
) -> Tensor:
    """Absolute vector norm loss: mean(||pred - target||_ord).

    Absolute counterpart to normalized_vector_norm_loss (relative). Computes
    the raw error norm without dividing by the target magnitude.

    **Differentiable**: ✅ Yes

    Shape:
        preds:  (B, ..., D) where D is vector dimension
        target: (B, ..., D) must match preds
        output: (,) scalar (default aggregator)

    Args:
        preds: Predicted vectors
        target: Ground truth vectors
        ord: Norm order (1=L1, 2=L2, float('inf')=Linf)
        dim: Vector dimension (default: -1)
        aggregator: Reduction function applied to per-sample norms (default: torch.mean)

    Returns:
        Aggregated absolute vector norm loss (differentiable)
    """
    if preds.dim() < 2:
        raise ValueError(f"Expected at least 2D tensors for vector operations, got {preds.dim()}D")
    error_vecs = compute_error_vectors(preds, target)
    error_norms = compute_vector_norm(error_vecs, ord=ord, dim=dim)
    return apply_aggregation(error_norms, aggregator)


def normalized_vector_norm_loss(
    preds: Tensor,
    target: Tensor,
    ord: int = 2,
    dim: int = -1,
    eps: float = 1e-8,
    aggregator: AggregatorFn = torch.mean,
) -> Tensor:
    """Relative vector norm loss: mean(||pred - target||_ord / ||target||_ord).

    Computes relative error normalized by target magnitude. Useful for
    vector predictions where scale varies across samples.

    **Differentiable**: ✅ Yes (tested with safe_divide eps=1e-8)

    Shape:
        preds:  (B, ..., D) where D is vector dimension
        target: (B, ..., D) must match preds
        output: (,) scalar (default aggregator)

    Args:
        preds: Predicted vectors
        target: Ground truth vectors
        ord: Norm order (1=L1, 2=L2, float('inf')=Linf)
        dim: Vector dimension (default: -1)
        eps: Numerical stability epsilon (default: 1e-8)
        aggregator: Reduction function applied to per-sample relative errors (default: torch.mean)

    Returns:
        Aggregated relative vector norm loss (differentiable)

    Note:
        Safe division with eps ensures gradient flow when ||target|| ≈ 0.
    """
    from .metrics.functional import normalized_vector_norm_error

    return normalized_vector_norm_error(
        preds, target, ord=ord, dim=dim, eps=eps, aggregator=aggregator
    )


def energy_norm_loss(
    preds: Tensor,
    target: Tensor,
    matrix: Tensor,
    aggregator: AggregatorFn = torch.mean,
) -> Tensor:
    """Absolute energy norm (A-norm) loss: mean(||pred - target||_A).

    The A-norm of a vector u is defined as ||u||_A = sqrt(u^T A u), where A
    is a positive (semi-)definite matrix. This loss penalises errors in the
    metric induced by A, not the standard Euclidean metric.

    When A = I (identity matrix) this reduces to the standard L2 vector norm loss.

    **Differentiable**: ✅ Yes (clamp(min=0) before sqrt for safety)

    Shape:
        preds:  (B, D)
        target: (B, D)
        matrix: (B, D, D) per-sample SPD matrix, or (D, D) shared matrix
        output: (,) scalar (default aggregator)

    Args:
        preds: Predicted vectors.
        target: Ground truth vectors.
        matrix: Positive (semi-)definite matrix — one per sample or shared.
            Typically a feature from the dataset (e.g. stiffness matrix).
        aggregator: Reduction function applied to per-sample norms (default: torch.mean)

    Returns:
        Aggregated absolute energy norm loss (differentiable)

    Examples:
        >>> preds = torch.randn(4, 8)
        >>> target = torch.randn(4, 8)
        >>> A = torch.eye(8).expand(4, -1, -1)  # identity → same as L2
        >>> loss = energy_norm_loss(preds, target, A)
    """
    error_vecs = compute_error_vectors(preds, target)
    per_sample_norms = compute_energy_norm(error_vecs, matrix)
    return apply_aggregation(per_sample_norms, aggregator)


def relative_energy_norm_loss(
    preds: Tensor,
    target: Tensor,
    matrix: Tensor,
    eps: float = 1e-8,
    aggregator: AggregatorFn = torch.mean,
) -> Tensor:
    """Relative energy norm loss: mean(||pred - target||_A / ||target||_A).

    Normalises the absolute energy norm error by the A-norm of the target,
    giving a dimensionless relative error. Analogous to the "Notay loss"
    used in preconditioner learning (without the PCG-specific semantics).

    **Differentiable**: ✅ Yes (eps prevents division by zero)

    Shape:
        preds:  (B, D)
        target: (B, D)
        matrix: (B, D, D) per-sample SPD matrix, or (D, D) shared matrix
        output: (,) scalar (default aggregator)

    Args:
        preds: Predicted vectors.
        target: Ground truth vectors.
        matrix: Positive (semi-)definite matrix — one per sample or shared.
        eps: Numerical stability epsilon for division when ||target||_A ≈ 0
            (default: 1e-8)
        aggregator: Reduction function applied to per-sample relative errors
            (default: torch.mean)

    Returns:
        Aggregated relative energy norm loss (differentiable)

    Examples:
        >>> preds = torch.randn(4, 8)
        >>> target = torch.randn(4, 8)
        >>> A = torch.eye(8)  # shared identity matrix
        >>> loss = relative_energy_norm_loss(preds, target, A)
    """
    error_vecs = compute_error_vectors(preds, target)
    error_norms = compute_energy_norm(error_vecs, matrix)
    target_norms = compute_energy_norm(target, matrix)
    return apply_aggregation(safe_divide(error_norms, target_norms, eps=eps), aggregator)


def normalized_mse(
    preds: Tensor,
    target: Tensor,
    normalization: Literal["variance", "range", "mean"] = "variance",
    eps: float = 1e-8,
) -> Tensor:
    """Normalized MSE - MSE scaled by target statistics.

    Provides scale-invariant error measurement. Useful when target
    magnitudes vary significantly across samples or features.

    Shape:
        preds:  (*) - any shape
        target: (*) - must match preds
        output: (,) - scalar

    Args:
        preds: Predicted values
        target: Ground truth values
        normalization: Normalization method:
            - 'variance': MSE / var(target)
            - 'range': MSE / (max(target) - min(target))^2
            - 'mean': MSE / mean(target)^2
        eps: Numerical stability epsilon (default: 1e-8)

    Returns:
        Scalar normalized MSE (differentiable)

    Note:
        Division by statistics with eps ensures differentiability.
    """
    mse_value = mse(preds, target)

    if normalization == "variance":
        normalizer = torch.var(target) + eps
    elif normalization == "range":
        normalizer = (torch.max(target) - torch.min(target)) ** 2 + eps
    elif normalization == "mean":
        normalizer = torch.mean(target) ** 2 + eps
    else:
        raise ValueError(
            f"Invalid normalization: {normalization}. Choose from: 'variance', 'range', 'mean'"
        )

    return mse_value / normalizer


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Type aliases
    "AggregatorFn",
    # Standard from torchmetrics.functional
    "mse",
    "mae",
    "msle",
    "mape",
    # Custom differentiable losses
    "huber_loss",
    "smooth_l1_loss",
    "log_cosh_loss",
    "quantile_loss",
    "normalized_mse",
    # Vector norm losses (absolute and relative pair)
    "vector_norm_loss",
    "normalized_vector_norm_loss",
    # Energy norm losses (absolute and relative pair)
    "energy_norm_loss",
    "relative_energy_norm_loss",
]
