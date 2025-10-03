"""Modern, composable metrics system following SOLID principles.

This module provides a comprehensive metrics framework with:
- Protocol-based interfaces for type safety
- Strategy patterns for composable aggregation and normalization
- Factory patterns for easy metric creation
- Template method patterns for consistent computation flow
- Function-like callable interface for intuitive usage

All metrics can be used both as methods and as callable functions:

Examples:
    >>> import torch
    >>> from dlkit.core.training.metrics import create_metric
    >>>
    >>> # Create metrics
    >>> mse = create_metric("mse")
    >>> mae = create_metric("mae")
    >>>
    >>> # Sample dataflow
    >>> predictions = torch.tensor([1.0, 2.0, 3.0])
    >>> targets = torch.tensor([1.1, 1.9, 3.1])
    >>>
    >>> # Both syntaxes work identically:
    >>> error1 = mse.compute(predictions, targets)  # Method syntax
    >>> error2 = mse(predictions, targets)  # Function syntax
    >>> assert torch.allclose(error1, error2)  # Same result
    >>>
    >>> # Functional programming style:
    >>> metrics = [mse, mae]
    >>> errors = [metric(predictions, targets) for metric in metrics]
    >>>
    >>> # Higher-order functions:
    >>> results = list(map(lambda m: m(predictions, targets), metrics))
"""

# Core protocol interfaces
from .protocols import IMetric, IAggregator, INormalizer, IMetricRegistry, IMetricFactory

# Base classes and patterns
from .base import BaseMetric, CompositeMetric, MetricDecorator

# Concrete metric implementations
from .implementations import (
    MeanSquaredErrorMetric,
    MeanAbsoluteErrorMetric,
    RootMeanSquaredErrorMetric,
    NormalizedVectorNormErrorMetric,
    MSEOverVarianceMetric,
    TemporalDerivativeMetric,
)

# Aggregation strategies
from .aggregators import (
    MeanAggregator,
    SumAggregator,
    VectorNormAggregator,
    StdAggregator,
    MEAN_AGGREGATOR,
    SUM_AGGREGATOR,
    L2_AGGREGATOR,
    L1_AGGREGATOR,
    STD_AGGREGATOR,
)

# Normalization strategies
from .normalizers import (
    VarianceNormalizer,
    StandardDeviationNormalizer,
    VectorNormNormalizer,
    NaiveForecastNormalizer,
    VARIANCE_NORMALIZER,
    STD_NORMALIZER,
    L2_NORM_NORMALIZER,
    L1_NORM_NORMALIZER,
    NAIVE_FORECAST_NORMALIZER,
)

# Registry and factory system
from .registry import (
    MetricRegistry,
    AggregatorRegistry,
    NormalizerRegistry,
    MetricFactory,
    get_global_metric_registry,
    get_global_aggregator_registry,
    get_global_normalizer_registry,
    get_global_metric_factory,
)


def create_metric(
    metric_type: str, aggregator: str = None, normalizer: str = None, **kwargs
) -> IMetric:
    """Create a metric using the global factory.

    Args:
        metric_type: Type of metric to create (mse, mae, rmse, etc.)
        aggregator: Name of aggregation strategy (mean, sum, l2_norm, etc.)
        normalizer: Name of normalization strategy (variance, std, etc.)
        **kwargs: Additional metric-specific parameters

    Returns:
        Configured metric instance

    Examples:
        >>> # Basic MSE metric
        >>> mse = create_metric("mse")
        >>>
        >>> # MAE with sum aggregation
        >>> mae_sum = create_metric("mae", aggregator="sum")
        >>>
        >>> # MSE with variance normalization
        >>> mse_normalized = create_metric("mse", normalizer="variance")
        >>>
        >>> # Custom parameters
        >>> metric = create_metric("mse", eps=1e-10, dim=0)
    """
    factory = get_global_metric_factory()
    return factory.create_metric(metric_type, aggregator, normalizer, **kwargs)


def create_normalized_vector_norm_error(
    vector_dim: int = -1, norm_ord: int = 2, aggregator: str = "mean", eps: float = 1e-8, **kwargs
) -> NormalizedVectorNormErrorMetric:
    """Create normalized vector norm error metric for 2D

    This metric computes the error between predicted and target vectors,
    where each vector is normalized by the target vector's norm. Perfect
    for measuring relative error in vector predictions.

    Args:
        vector_dim: Dimension along which vectors are defined (default: -1)
        norm_ord: Order of the norm (1, 2, inf, etc.) (default: 2)
        aggregator: Name of aggregation strategy (default: "mean")
        eps: Small value for numerical stability (default: 1e-8)
        **kwargs: Additional parameters

    Returns:
        Configured NormalizedVectorNormErrorMetric instance

    Examples:
        >>> import torch
        >>>
        >>> # Create metric for 2D vectors with L2 norm
        >>> metric = create_normalized_vector_norm_error()
        >>>
        >>> # Sample 2D vector dataflow (each row is a vector)
        >>> predictions = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        >>> targets = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
        >>>
        >>> # Compute normalized error
        >>> error = metric.compute(predictions, targets)
        >>> print(f"Normalized vector error: {error:.4f}")
        >>>
        >>> # L1 norm with sum aggregation
        >>> l1_metric = create_normalized_vector_norm_error(norm_ord=1, aggregator="sum")
    """
    factory = get_global_metric_factory()
    return factory.create_normalized_vector_norm_error(
        vector_dim=vector_dim, norm_ord=norm_ord, aggregator=aggregator, eps=eps, **kwargs
    )


def create_composite_metric(
    name: str, metrics: list[IMetric], weights: list[float] = None
) -> CompositeMetric:
    """Create a composite metric from multiple metrics.

    Args:
        name: Name for the composite metric
        metrics: List of metrics to combine
        weights: Optional weights for weighted combination

    Returns:
        CompositeMetric instance

    Examples:
        >>> # Equal weight combination
        >>> mse = create_metric("mse")
        >>> mae = create_metric("mae")
        >>> composite = create_composite_metric("mse_mae", [mse, mae])
        >>>
        >>> # Weighted combination (70% MSE, 30% MAE)
        >>> weighted = create_composite_metric("weighted_error", [mse, mae], weights=[0.7, 0.3])
    """
    import torch

    weights_tensor = torch.tensor(weights) if weights else None
    return CompositeMetric(name, metrics, weights_tensor)


__all__ = [
    # Protocol interfaces
    "IMetric",
    "IAggregator",
    "INormalizer",
    "IMetricRegistry",
    "IMetricFactory",
    # Base classes
    "BaseMetric",
    "CompositeMetric",
    "MetricDecorator",
    # Metric implementations
    "MeanSquaredErrorMetric",
    "MeanAbsoluteErrorMetric",
    "RootMeanSquaredErrorMetric",
    "NormalizedVectorNormErrorMetric",
    "MSEOverVarianceMetric",
    "TemporalDerivativeMetric",
    # Aggregators
    "MeanAggregator",
    "SumAggregator",
    "VectorNormAggregator",
    "StdAggregator",
    "MEAN_AGGREGATOR",
    "SUM_AGGREGATOR",
    "L2_AGGREGATOR",
    "L1_AGGREGATOR",
    "STD_AGGREGATOR",
    # Normalizers
    "VarianceNormalizer",
    "StandardDeviationNormalizer",
    "VectorNormNormalizer",
    "NaiveForecastNormalizer",
    "VARIANCE_NORMALIZER",
    "STD_NORMALIZER",
    "L2_NORM_NORMALIZER",
    "L1_NORM_NORMALIZER",
    "NAIVE_FORECAST_NORMALIZER",
    # Registry and factory
    "MetricRegistry",
    "AggregatorRegistry",
    "NormalizerRegistry",
    "MetricFactory",
    "get_global_metric_registry",
    "get_global_aggregator_registry",
    "get_global_normalizer_registry",
    "get_global_metric_factory",
    # Convenience functions
    "create_metric",
    "create_normalized_vector_norm_error",
    "create_composite_metric",
]
