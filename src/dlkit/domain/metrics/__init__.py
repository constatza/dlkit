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
    For loss functions, see dlkit.domain.losses which provides
    differentiable implementations suitable for backpropagation.

Examples:
    >>> import torch
    >>> from dlkit.domain.metrics import (
    ...     MeanSquaredError,  # Standard metric from torchmetrics
    ...     NormalizedVectorNormError,  # Custom metric
    ... )
    >>> from torchmetrics import MetricCollection
    >>>
    >>> # Use custom metrics in MetricCollection
    >>> metrics = MetricCollection(
    ...     {
    ...         "mse": MeanSquaredError(),
    ...         "norm_error": NormalizedVectorNormError(norm_ord=2),
    ...     }
    ... )
    >>>
    >>> preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> target = torch.tensor([[1.1, 1.9], [3.1, 3.9]])
    >>>
    >>> metrics.update(preds, target)
    >>> results = metrics.compute()
    >>> # Results can be logged to MLflow!
    >>>
    >>> # Functional interface (for advanced users)
    >>> from dlkit.domain.metrics.functional import (
    ...     normalized_vector_norm_error,
    ...     temporal_derivative_error,
    ... )
    >>> error = normalized_vector_norm_error(preds, target, ord=2, dim=-1)
"""

# ============================================================================
# STANDARD METRICS (delegated to external torchmetrics)
# ============================================================================

# ============================================================================
# UTILITIES
# ============================================================================
from .collect import MetricsPayload
from .compat import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanSquaredLogError,
    R2Score,
)

# ============================================================================
# FUNCTIONAL INTERFACE (for advanced users)
# ============================================================================
from .functional import (
    # Type aliases
    AggregatorFn,
    apply_aggregation,
    compute_energy_norm,
    # Composable primitives
    compute_error_vectors,
    # Energy norm primitives
    compute_quadratic_form,
    # Temporal metrics
    compute_temporal_derivative,
    compute_vector_norm,
    first_derivative_error,
    normalized_l1_error,
    normalized_l2_error,
    normalized_linf_error,
    # Vector metrics
    normalized_vector_norm_error,
    safe_divide,
    second_derivative_error,
    temporal_derivative_error,
)

# ============================================================================
# CUSTOM TORCHMETRICS WRAPPERS (our specialized metrics)
# ============================================================================
from .torchmetrics_wrappers import (
    AbsoluteVectorNormError,
    EnergyNormError,
    NormalizedVectorNormError,
    RelativeEnergyNormError,
    TemporalDerivativeError,
)

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Type aliases
    "AggregatorFn",
    "MetricsPayload",
    # Standard metrics (from torchmetrics)
    "MeanSquaredError",
    "MeanAbsoluteError",
    "MeanSquaredLogError",
    "MeanAbsolutePercentageError",
    "R2Score",
    # Custom torchmetrics wrappers
    "NormalizedVectorNormError",
    "TemporalDerivativeError",
    "AbsoluteVectorNormError",
    "EnergyNormError",
    "RelativeEnergyNormError",
    # Functional interface
    "compute_error_vectors",
    "compute_vector_norm",
    "safe_divide",
    "apply_aggregation",
    "normalized_vector_norm_error",
    "normalized_l1_error",
    "normalized_l2_error",
    "normalized_linf_error",
    "compute_quadratic_form",
    "compute_energy_norm",
    "compute_temporal_derivative",
    "temporal_derivative_error",
    "first_derivative_error",
    "second_derivative_error",
]
