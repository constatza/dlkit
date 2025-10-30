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
) -> Tensor:
    """Huber loss - smooth combination of L1 and L2 loss.

    Less sensitive to outliers than MSE. Uses L2 loss for small errors
    and L1 loss for large errors.

    Shape:
        preds:  (*) - any shape
        target: (*) - must match preds
        output: (,) - scalar

    Args:
        preds: Predicted values
        target: Ground truth values
        delta: Threshold for switching from L2 to L1 (default: 1.0)

    Returns:
        Scalar Huber loss (differentiable)

    References:
        Huber, P. J. (1964). Robust Estimation of a Location Parameter.
    """
    error = preds - target
    abs_error = torch.abs(error)

    # Quadratic for small errors, linear for large errors
    quadratic = 0.5 * error ** 2
    linear = delta * (abs_error - 0.5 * delta)

    return torch.mean(torch.where(abs_error <= delta, quadratic, linear))


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


def log_cosh_loss(preds: Tensor, target: Tensor) -> Tensor:
    """Log-cosh loss - smooth approximation of MAE.

    Less sensitive to outliers than MSE, smoother than MAE.
    Approximately equal to (x^2 / 2) for small x, |x| for large x.

    Shape:
        preds:  (*) - any shape
        target: (*) - must match preds
        output: (,) - scalar

    Args:
        preds: Predicted values
        target: Ground truth values

    Returns:
        Scalar log-cosh loss (differentiable)

    References:
        https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
    """
    error = preds - target
    return torch.mean(torch.log(torch.cosh(error)))


def quantile_loss(
    preds: Tensor,
    target: Tensor,
    quantile: float = 0.5,
) -> Tensor:
    """Quantile loss (pinball loss) for quantile regression.

    When quantile=0.5, equivalent to MAE. Useful for predicting
    different quantiles of the target distribution.

    Shape:
        preds:  (*) - any shape
        target: (*) - must match preds
        output: (,) - scalar

    Args:
        preds: Predicted values
        target: Ground truth values
        quantile: Desired quantile in [0, 1] (default: 0.5 = median)

    Returns:
        Scalar quantile loss (differentiable)

    Raises:
        ValueError: If quantile not in [0, 1]
    """
    if not 0 <= quantile <= 1:
        raise ValueError(f"Quantile must be in [0, 1], got {quantile}")

    error = target - preds
    # Asymmetric penalty: heavier for underestimation (pred < target) when quantile > 0.5
    return torch.mean(torch.where(error >= 0, quantile * error, (quantile - 1) * error))


# ============================================================================
# NORMALIZED LOSSES
# ============================================================================


def normalized_vector_norm_loss(
    preds: Tensor,
    target: Tensor,
    ord: int = 2,
    dim: int = -1,
    eps: float = 1e-8,
) -> Tensor:
    """Normalized vector norm loss: mean(||pred - target|| / ||target||).

    Computes relative error normalized by target magnitude. Useful for
    vector predictions where scale varies across samples.

    **Differentiable**: ✅ Yes (tested with safe_divide eps=1e-8)

    Shape:
        preds:  (B, ..., D) where D is vector dimension
        target: (B, ..., D) must match preds
        output: (,) scalar

    Args:
        preds: Predicted vectors
        target: Ground truth vectors
        ord: Norm order (1=L1, 2=L2, float('inf')=Linf)
        dim: Vector dimension (default: -1)
        eps: Numerical stability epsilon (default: 1e-8)

    Returns:
        Scalar normalized vector norm loss (differentiable)

    Note:
        Imported from metrics.functional and verified differentiable.
        Safe division with eps ensures gradient flow.
    """
    from .metrics.functional import normalized_vector_norm_error

    return normalized_vector_norm_error(
        preds, target, ord=ord, dim=dim, eps=eps, aggregator=torch.mean
    )


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
            f"Invalid normalization: {normalization}. "
            f"Choose from: 'variance', 'range', 'mean'"
        )

    return mse_value / normalizer


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
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
    "normalized_vector_norm_loss",
]
