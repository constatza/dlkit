"""Functional metric implementations following pure functional programming principles.

This module provides composable, pure functions for computing custom metrics.
All functions are stateless and side-effect free, enabling easy composition
via functools and higher-order function patterns.

Design Principles:
    - Pure functions (no side effects)
    - Composition via functools.partial
    - Aggregators as first-class Callable parameters
    - Clear shape contracts in every docstring
    - Readable Python (no over-engineered FP abstractions)

Structure:
    1. Composable Building Blocks - Pure primitives
    2. Vector Metrics - Metrics for multi-dimensional vectors
    3. Energy Norm Primitives - A-norm / quadratic form building blocks
    4. Temporal Metrics - Metrics with temporal/sequence dimensions (3D inputs)
    5. Update/Compute Split - For torchmetrics wrappers

Shape Notation:
    B = Batch size
    D = Vector/feature dimension
    T = Temporal/sequence dimension
    (,) = Scalar tensor
"""

from functools import partial
from typing import Callable, TypeAlias

import torch
from torch import Tensor

# ============================================================================
# TYPE ALIASES
# ============================================================================

AggregatorFn: TypeAlias = Callable[[Tensor], Tensor]
"""Reduction function that maps a tensor of per-sample values to a scalar.

Examples:
    torch.mean, torch.sum, partial(torch.mean, dim=0)
"""


# ============================================================================
# 1. COMPOSABLE BUILDING BLOCKS
# ============================================================================


def compute_error_vectors(preds: Tensor, target: Tensor) -> Tensor:
    """Compute raw error vectors: predictions - targets.

    This is the fundamental error computation used by all metrics.

    Shape:
        preds:  (B, ..., D) - arbitrary shape
        target: (B, ..., D) - must match preds
        output: (B, ..., D) - element-wise errors

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    Returns:
        Element-wise error tensor

    Raises:
        ValueError: If shapes don't match
    """
    if preds.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} != target {target.shape}"
        )
    return preds - target


def compute_vector_norm(tensor: Tensor, ord: int = 2, dim: int = -1) -> Tensor:
    """Compute vector norm along specified dimension.

    Shape:
        input:  (B, ..., D, ...) - arbitrary dimensions
        output: (B, ..., ...) - dimension D is reduced

    Args:
        tensor: Input tensor
        ord: Norm order (1=L1, 2=L2, float('inf')=Linf)
        dim: Dimension to compute norm over

    Returns:
        Norm values with specified dimension reduced

    Raises:
        ValueError: If dimension is out of bounds
    """
    if abs(dim) > tensor.dim():
        raise ValueError(
            f"Dimension {dim} out of bounds for {tensor.dim()}D tensor"
        )
    return torch.linalg.vector_norm(tensor, ord=ord, dim=dim)


def safe_divide(numerator: Tensor, denominator: Tensor, eps: float = 1e-8) -> Tensor:
    """Safe division with epsilon to prevent division by zero.

    Shape:
        numerator:   (B, ...) - arbitrary shape
        denominator: (B, ...) - must match numerator or be broadcastable
        output:      (B, ...) - same as numerator

    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        eps: Small epsilon for numerical stability

    Returns:
        Safely divided tensor
    """
    return numerator / (denominator + eps)


def apply_aggregation(
    tensor: Tensor, aggregator: Callable[[Tensor], Tensor] = torch.mean
) -> Tensor:
    """Apply aggregation function to reduce tensor.

    Shape:
        input:  (B, ...) - arbitrary shape
        output: (,) or (B, ...) - depends on aggregator function

    Args:
        tensor: Input tensor to aggregate
        aggregator: Reduction function (default: torch.mean)
            Should accept a single Tensor and return reduced Tensor.
            For aggregators with kwargs, use functools.partial.

    Returns:
        Aggregated tensor

    Examples:
        >>> # Simple aggregation
        >>> apply_aggregation(errors, torch.mean)
        >>>
        >>> # Aggregation with dimension
        >>> from functools import partial
        >>> apply_aggregation(errors, partial(torch.mean, dim=0))
    """
    return aggregator(tensor)


# ============================================================================
# 2. VECTOR METRICS
# ============================================================================
# Metrics for multi-dimensional vectors where each sample is a vector
# and we compute error/similarity between vectors.
# ============================================================================


def normalized_vector_norm_error(
    preds: Tensor,
    target: Tensor,
    ord: int = 2,
    dim: int = -1,
    eps: float = 1e-8,
    aggregator: Callable[[Tensor], Tensor] = torch.mean,
) -> Tensor:
    """Compute normalized vector norm error: ||pred - target|| / ||target||.

    This metric normalizes error by target magnitude, providing relative
    error measurement. Particularly useful for vector predictions where
    magnitude varies significantly across samples.

    Shape Contract:
        preds:  (B, ..., D) where D is the vector dimension
        target: (B, ..., D) must match preds exactly
        output: (,) if aggregator reduces to scalar, else (B, ...)

    Args:
        preds: Predicted vectors
        target: Ground truth vectors
        ord: Norm order (1=L1/Manhattan, 2=L2/Euclidean, float('inf')=Linf)
        dim: Vector dimension to compute norm over (default: -1)
        eps: Numerical stability epsilon for division (default: 1e-8)
        aggregator: Function to aggregate per-sample errors (default: torch.mean)

    Returns:
        Normalized error - scalar if aggregator reduces, else per-sample errors

    Raises:
        ValueError: If shapes don't match or dimension out of bounds
        ValueError: If tensor has less than 2 dimensions

    Examples:
        >>> # 2D vectors (e.g., velocity predictions)
        >>> preds = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        >>> target = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
        >>> error = normalized_vector_norm_error(preds, target)
        >>> # error ≈ mean([0.7071, 1.0]) ≈ 0.8536
        >>>
        >>> # L1 norm variant with partial
        >>> from functools import partial
        >>> l1_error_fn = partial(normalized_vector_norm_error, ord=1)
        >>> error_l1 = l1_error_fn(preds, target)
    """
    # Validation
    if preds.dim() < 2:
        raise ValueError(
            f"Expected at least 2D tensors for vector operations, got {preds.dim()}D"
        )

    error_vecs = compute_error_vectors(preds, target)
    error_norms = compute_vector_norm(error_vecs, ord=ord, dim=dim)
    target_norms = compute_vector_norm(target, ord=ord, dim=dim)
    normalized = safe_divide(error_norms, target_norms, eps=eps)
    return apply_aggregation(normalized, aggregator)


# Convenience partials for common norm orders
normalized_l1_error = partial(normalized_vector_norm_error, ord=1)
normalized_l2_error = partial(normalized_vector_norm_error, ord=2)
normalized_linf_error = partial(normalized_vector_norm_error, ord=float("inf"))


# ============================================================================
# 3. ENERGY NORM PRIMITIVES
# ============================================================================
# Building blocks for the A-norm (energy norm) ||u||_A = sqrt(u^T A u).
# These primitives operate per-sample with no reduction.
#
# SHAPE ASSUMPTIONS:
#   - vector: (B, D) — one D-dimensional vector per sample
#   - matrix: (B, D, D) per-sample SPD matrix, OR (D, D) shared matrix
#   - output: (B,) — one scalar per sample (pre-aggregation)
# ============================================================================


def compute_quadratic_form(vector: Tensor, matrix: Tensor) -> Tensor:
    """Compute per-sample quadratic form v^T A v.

    Supports both per-sample and shared matrices via broadcasting.

    Shape Contract:
        vector: (B, D)
        matrix: (B, D, D) per-sample, or (D, D) shared (broadcast over B)
        output: (B,) — one scalar per sample

    Args:
        vector: Batch of vectors.
        matrix: Positive (semi-)definite matrix per sample or shared.

    Returns:
        Per-sample quadratic form values.

    Examples:
        >>> v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # (2, 2)
        >>> A = torch.eye(2)  # (2, 2) — identity (shared)
        >>> compute_quadratic_form(v, A)  # tensor([1., 1.])
    """
    # Av: (B, D) — handles both (B, D, D) and (D, D) via matmul broadcasting
    Av = torch.matmul(matrix, vector.unsqueeze(-1)).squeeze(-1)
    return (vector * Av).sum(dim=-1)


def compute_energy_norm(vector: Tensor, matrix: Tensor) -> Tensor:
    """Compute per-sample A-norm sqrt(v^T A v).

    Uses clamp(min=0) before sqrt for numerical safety when A is near-singular.

    Shape Contract:
        vector: (B, D)
        matrix: (B, D, D) per-sample, or (D, D) shared (broadcast over B)
        output: (B,) — one norm value per sample

    Args:
        vector: Batch of vectors.
        matrix: Positive (semi-)definite matrix per sample or shared.

    Returns:
        Per-sample A-norm values.

    Examples:
        >>> v = torch.tensor([[3.0, 4.0]])  # (1, 2)
        >>> A = torch.eye(2)               # identity → ||v||_I = ||v||_2 = 5.0
        >>> compute_energy_norm(v, A)      # tensor([5.])
    """
    return torch.sqrt(compute_quadratic_form(vector, matrix).clamp(min=0))


# ============================================================================
# 4. TEMPORAL METRICS
# ============================================================================
# Metrics that operate on sequences/time series with a temporal dimension.
# These metrics compute derivatives or differences along time axis.
#
# SHAPE ASSUMPTIONS FOR TEMPORAL METRICS:
#   - Input shape: (B, T, D)
#       B = Batch size
#       T = Temporal/sequence dimension (number of time steps)
#       D = Feature/vector dimension at each time step
#
#   - Output shape: (B, T-n, D) after nth derivative
#       Temporal dimension is reduced by n (derivative order)
#
#   - Requirements:
#       * Must have exactly 3 dimensions
#       * T must be >= n+1 (need at least n+1 points for nth derivative)
#       * derivative_dim must be valid (typically 1 for middle dimension)
# ============================================================================


def compute_temporal_derivative(
    tensor: Tensor, n: int = 1, derivative_dim: int = 1
) -> Tensor:
    """Compute nth-order finite difference approximation of derivative.

    Uses torch.diff to compute discrete differences along temporal dimension.
    This approximates derivatives for sequence data.

    Shape Contract:
        input:  (B, T, D) - 3D tensor with temporal dimension
        output: (B, T-n, D) - temporal dimension reduced by n

    Args:
        tensor: Input sequence tensor with shape (B, T, D)
        n: Derivative order (1=velocity, 2=acceleration, etc.)
        derivative_dim: Temporal dimension index (default: 1 for middle dim)

    Returns:
        Finite difference along temporal dimension

    Raises:
        ValueError: If tensor is not 3D
        ValueError: If temporal dimension T < n+1

    Examples:
        >>> # Position sequence (batch=1, time=4, features=1)
        >>> position = torch.tensor([[[0.0], [1.0], [3.0], [6.0]]])  # (1, 4, 1)
        >>> velocity = compute_temporal_derivative(position, n=1, derivative_dim=1)
        >>> # velocity shape: (1, 3, 1), values: [[[1.0], [2.0], [3.0]]]
        >>>
        >>> # Acceleration (2nd derivative)
        >>> accel = compute_temporal_derivative(position, n=2, derivative_dim=1)
        >>> # accel shape: (1, 2, 1), values: [[[1.0], [1.0]]]
    """
    # Validation: must be 3D
    if tensor.dim() != 3:
        raise ValueError(
            f"Temporal metrics require 3D input (B, T, D), got {tensor.dim()}D"
        )

    # Validation: temporal dimension must be sufficient
    temporal_size = tensor.size(derivative_dim)
    if temporal_size < n + 1:
        raise ValueError(
            f"Temporal dimension T={temporal_size} too small for "
            f"{n}th derivative (need T >= {n + 1})"
        )

    return torch.diff(tensor, n=n, dim=derivative_dim)


def temporal_derivative_error(
    preds: Tensor,
    target: Tensor,
    n: int = 1,
    derivative_dim: int = 1,
    aggregator: Callable[[Tensor], Tensor] = torch.mean,
) -> Tensor:
    """Compute temporal derivative error: mean squared error of nth derivatives.

    Measures how well predictions match target dynamics (velocity, acceleration, etc.)
    rather than just static values. Useful for physics-based predictions.

    Shape Contract:
        preds:  (B, T, D) - 3D tensor with batch, time, features
        target: (B, T, D) - must match preds exactly
        output: (,) if aggregator reduces to scalar, else (B, T-n, D)

    Temporal Dimension Requirements:
        - Must have exactly 3 dimensions (B, T, D)
        - Must have T >= n+1 time steps
        - derivative_dim must be valid (typically 1 for middle dimension)
        - After n differences, temporal dim becomes T-n

    Args:
        preds: Predicted sequence with shape (B, T, D)
        target: Ground truth sequence with shape (B, T, D)
        n: Derivative order (1=first derivative/velocity, 2=second/acceleration, etc.)
        derivative_dim: Index of temporal dimension (default: 1 for middle dim)
        aggregator: Function to aggregate errors (default: torch.mean)

    Returns:
        Squared derivative error - scalar if aggregator reduces

    Raises:
        ValueError: If inputs are not 3D
        ValueError: If T < n+1
        ValueError: If shapes don't match

    Examples:
        >>> # Predict trajectory dynamics (batch=1, time=4, features=2)
        >>> preds = torch.tensor([[[0.0, 0.0], [1.0, 0.5], [2.5, 1.5], [4.0, 2.0]]])
        >>> target = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        >>>
        >>> # First derivative error (velocity error)
        >>> vel_error = temporal_derivative_error(preds, target, n=1)
        >>> # Compares velocity: [[1.0,0.5],[1.5,1.0],[1.5,0.5]] vs [[1.0,0.0],[1.0,0.0],[1.0,0.0]]
        >>>
        >>> # Second derivative error (acceleration error)
        >>> accel_error = temporal_derivative_error(preds, target, n=2)
        >>> # Compares acceleration across time
    """
    # Validate inputs (compute_temporal_derivative will validate dimensions)
    error = compute_error_vectors(preds, target)
    derivative_error = compute_temporal_derivative(error, n=n, derivative_dim=derivative_dim)
    squared_derivative = torch.pow(derivative_error, 2)
    return apply_aggregation(squared_derivative, aggregator)


# Convenience partials for common derivative orders
first_derivative_error = partial(temporal_derivative_error, n=1)
second_derivative_error = partial(temporal_derivative_error, n=2)


# ============================================================================
# 5. UPDATE/COMPUTE SPLIT FOR TORCHMETRICS WRAPPERS
# ============================================================================
# These functions support the torchmetrics update()/compute() pattern
# by separating state accumulation from final metric computation.
# ============================================================================


def _normalized_vector_norm_update(
    preds: Tensor, target: Tensor, ord: int, dim: int, eps: float
) -> Tensor:
    """Compute per-sample normalized errors without aggregation.

    Used by torchmetrics wrapper to accumulate state across batches.

    Shape:
        preds:  (B, ..., D)
        target: (B, ..., D)
        output: (B, ...) - one normalized error per vector

    Args:
        preds: Predicted vectors
        target: Ground truth vectors
        ord: Norm order
        dim: Vector dimension
        eps: Numerical stability epsilon

    Returns:
        Per-sample normalized errors (not aggregated)
    """
    if preds.dim() < 2:
        raise ValueError(
            f"Expected at least 2D tensors for vector operations, got {preds.dim()}D"
        )

    error_vecs = compute_error_vectors(preds, target)
    error_norms = compute_vector_norm(error_vecs, ord=ord, dim=dim)
    target_norms = compute_vector_norm(target, ord=ord, dim=dim)
    return safe_divide(error_norms, target_norms, eps=eps)


def _normalized_vector_norm_compute(sum_errors: Tensor, total: int) -> Tensor:
    """Compute final mean from accumulated state.

    Shape:
        sum_errors: (,) - scalar sum of all errors
        total:      int - total number of samples
        output:     (,) - scalar mean

    Args:
        sum_errors: Accumulated sum of normalized errors
        total: Total number of error values

    Returns:
        Mean normalized error
    """
    return sum_errors / total


def _temporal_derivative_update(
    preds: Tensor, target: Tensor, n: int, derivative_dim: int
) -> Tensor:
    """Compute squared derivative errors without aggregation.

    Used by torchmetrics wrapper to accumulate state across batches.

    Shape:
        preds:  (B, T, D) - 3D temporal tensor
        target: (B, T, D) - 3D temporal tensor
        output: (B, T-n, D) - squared derivative errors

    Args:
        preds: Predicted sequence (B, T, D)
        target: Ground truth sequence (B, T, D)
        n: Derivative order
        derivative_dim: Temporal dimension (typically 1)

    Returns:
        Per-sample squared derivative errors (not aggregated)

    Raises:
        ValueError: If not 3D or T < n+1
    """
    error = compute_error_vectors(preds, target)
    derivative_error = compute_temporal_derivative(error, n=n, derivative_dim=derivative_dim)
    return torch.pow(derivative_error, 2)


def _temporal_derivative_compute(sum_squared_errors: Tensor, total: int) -> Tensor:
    """Compute final mean squared derivative error from accumulated state.

    Shape:
        sum_squared_errors: (,) - scalar sum
        total:              int
        output:             (,) - scalar mean

    Args:
        sum_squared_errors: Accumulated sum of squared derivative errors
        total: Total number of error values

    Returns:
        Mean squared derivative error
    """
    return sum_squared_errors / total


def _absolute_vector_norm_update(
    preds: Tensor, target: Tensor, ord: int, dim: int
) -> Tensor:
    """Compute per-sample absolute vector norm errors without aggregation.

    Used by torchmetrics wrapper to accumulate state across batches.

    Shape:
        preds:  (B, ..., D)
        target: (B, ..., D)
        output: (B, ...) — one norm value per vector

    Args:
        preds: Predicted vectors
        target: Ground truth vectors
        ord: Norm order (1=L1, 2=L2, float('inf')=Linf)
        dim: Vector dimension

    Returns:
        Per-sample absolute vector norm errors (not aggregated)
    """
    if preds.dim() < 2:
        raise ValueError(
            f"Expected at least 2D tensors for vector operations, got {preds.dim()}D"
        )
    error_vecs = compute_error_vectors(preds, target)
    return compute_vector_norm(error_vecs, ord=ord, dim=dim)


def _absolute_vector_norm_compute(sum_norms: Tensor, total: int) -> Tensor:
    """Compute final mean absolute vector norm error from accumulated state.

    Shape:
        sum_norms: (,) - scalar sum of norms
        total:     int
        output:    (,) - scalar mean

    Args:
        sum_norms: Accumulated sum of per-sample norm errors
        total: Total number of samples

    Returns:
        Mean absolute vector norm error
    """
    return sum_norms / total


def _energy_norm_update(
    preds: Tensor, target: Tensor, matrix: Tensor
) -> Tensor:
    """Compute per-sample absolute energy norm errors without aggregation.

    Used by torchmetrics wrapper to accumulate state across batches.

    Shape:
        preds:  (B, D)
        target: (B, D)
        matrix: (B, D, D) or (D, D) shared
        output: (B,) — one A-norm value per sample

    Args:
        preds: Predicted vectors
        target: Ground truth vectors
        matrix: Positive (semi-)definite matrix per sample or shared

    Returns:
        Per-sample absolute energy norm errors (not aggregated)
    """
    error_vecs = compute_error_vectors(preds, target)
    return compute_energy_norm(error_vecs, matrix)


def _energy_norm_compute(sum_norms: Tensor, total: int) -> Tensor:
    """Compute final mean energy norm error from accumulated state.

    Shape:
        sum_norms: (,) - scalar sum
        total:     int
        output:    (,) - scalar mean

    Args:
        sum_norms: Accumulated sum of per-sample energy norm errors
        total: Total number of samples

    Returns:
        Mean absolute energy norm error
    """
    return sum_norms / total


def _relative_energy_norm_update(
    preds: Tensor, target: Tensor, matrix: Tensor, eps: float = 1e-8
) -> Tensor:
    """Compute per-sample relative energy norm errors without aggregation.

    Used by torchmetrics wrapper to accumulate state across batches.

    Shape:
        preds:  (B, D)
        target: (B, D)
        matrix: (B, D, D) or (D, D) shared
        output: (B,) — one relative error per sample

    Args:
        preds: Predicted vectors
        target: Ground truth vectors
        matrix: Positive (semi-)definite matrix per sample or shared
        eps: Numerical stability epsilon for division

    Returns:
        Per-sample relative energy norm errors (not aggregated)
    """
    error_vecs = compute_error_vectors(preds, target)
    error_norms = compute_energy_norm(error_vecs, matrix)
    target_norms = compute_energy_norm(target, matrix)
    return safe_divide(error_norms, target_norms, eps=eps)


def _relative_energy_norm_compute(sum_norms: Tensor, total: int) -> Tensor:
    """Compute final mean relative energy norm error from accumulated state.

    Shape:
        sum_norms: (,) - scalar sum
        total:     int
        output:    (,) - scalar mean

    Args:
        sum_norms: Accumulated sum of per-sample relative energy norm errors
        total: Total number of samples

    Returns:
        Mean relative energy norm error
    """
    return sum_norms / total


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Type aliases
    "AggregatorFn",
    # Composable primitives
    "compute_error_vectors",
    "compute_vector_norm",
    "safe_divide",
    "apply_aggregation",
    # Vector metrics
    "normalized_vector_norm_error",
    "normalized_l1_error",
    "normalized_l2_error",
    "normalized_linf_error",
    # Energy norm primitives
    "compute_quadratic_form",
    "compute_energy_norm",
    # Temporal metrics
    "compute_temporal_derivative",
    "temporal_derivative_error",
    "first_derivative_error",
    "second_derivative_error",
    # Update/compute split (for torchmetrics)
    "_normalized_vector_norm_update",
    "_normalized_vector_norm_compute",
    "_absolute_vector_norm_update",
    "_absolute_vector_norm_compute",
    "_energy_norm_update",
    "_energy_norm_compute",
    "_relative_energy_norm_update",
    "_relative_energy_norm_compute",
    "_temporal_derivative_update",
    "_temporal_derivative_compute",
]
