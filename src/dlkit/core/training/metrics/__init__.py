"""Modern metrics system with functional core and torchmetrics integration.

This module provides a comprehensive metrics framework with:
    - Functional core: Pure, composable metric functions
    - TorchMetrics wrappers: Stateful wrappers for MetricCollection and MLflow
    - Standard metrics delegation: Aliases to external torchmetrics library
    - Loss compatibility: Functional metrics can be used as differentiable losses

Architecture:
    functional.py            - Pure functional implementations (metric-specific)
    ../functional.py         - Shared loss/metric functions (differentiable)
    torchmetrics_wrappers.py - Custom torchmetrics.Metric classes
    compat.py                - Delegates to external torchmetrics

All custom metrics are compatible with:
    - torchmetrics.MetricCollection
    - MLflow logging
    - Distributed training (DDP)
    - PyTorch Lightning

Note:
    For loss functions, see dlkit.core.training.functional which provides
    differentiable implementations suitable for backpropagation.

Examples:
    >>> import torch
    >>> from dlkit.core.training.metrics import (
    ...     MeanSquaredError,  # Standard metric from torchmetrics
    ...     NormalizedVectorNormError,  # Custom metric
    ... )
    >>> from torchmetrics import MetricCollection
    >>>
    >>> # Use custom metrics in MetricCollection
    >>> metrics = MetricCollection({
    ...     'mse': MeanSquaredError(),
    ...     'norm_error': NormalizedVectorNormError(norm_ord=2),
    ... })
    >>>
    >>> preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> target = torch.tensor([[1.1, 1.9], [3.1, 3.9]])
    >>>
    >>> metrics.update(preds, target)
    >>> results = metrics.compute()
    >>> # Results can be logged to MLflow!
    >>>
    >>> # Functional interface (for advanced users)
    >>> from dlkit.core.training.metrics.functional import (
    ...     normalized_vector_norm_error,
    ...     temporal_derivative_error,
    ... )
    >>> error = normalized_vector_norm_error(preds, target, ord=2, dim=-1)
"""

# ============================================================================
# STANDARD METRICS (delegated to external torchmetrics)
# ============================================================================

from .compat import (
    MeanSquaredError,
    MeanAbsoluteError,
    MeanSquaredLogError,
    MeanAbsolutePercentageError,
    R2Score,
)

# ============================================================================
# CUSTOM TORCHMETRICS WRAPPERS (our specialized metrics)
# ============================================================================

from .torchmetrics_wrappers import (
    NormalizedVectorNormError,
    TemporalDerivativeError,
)

# ============================================================================
# FUNCTIONAL INTERFACE (for advanced users)
# ============================================================================

from .functional import (
    # Composable primitives
    compute_error_vectors,
    compute_vector_norm,
    safe_divide,
    apply_aggregation,
    # Vector metrics
    normalized_vector_norm_error,
    normalized_l1_error,
    normalized_l2_error,
    normalized_linf_error,
    # Temporal metrics
    compute_temporal_derivative,
    temporal_derivative_error,
    first_derivative_error,
    second_derivative_error,
)


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Standard metrics (from torchmetrics)
    "MeanSquaredError",
    "MeanAbsoluteError",
    "MeanSquaredLogError",
    "MeanAbsolutePercentageError",
    "R2Score",
    # Custom torchmetrics wrappers
    "NormalizedVectorNormError",
    "TemporalDerivativeError",
    # Functional interface
    "compute_error_vectors",
    "compute_vector_norm",
    "safe_divide",
    "apply_aggregation",
    "normalized_vector_norm_error",
    "normalized_l1_error",
    "normalized_l2_error",
    "normalized_linf_error",
    "compute_temporal_derivative",
    "temporal_derivative_error",
    "first_derivative_error",
    "second_derivative_error",
]
