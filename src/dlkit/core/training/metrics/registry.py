"""Metric registry and factory implementation following Registry Pattern."""

import threading

from .protocols import IMetric, IAggregator, INormalizer
from .implementations import (
    MeanSquaredErrorMetric,
    MeanAbsoluteErrorMetric,
    RootMeanSquaredErrorMetric,
    NormalizedVectorNormErrorMetric,
    MSEOverVarianceMetric,
    TemporalDerivativeMetric,
)
from .aggregators import (
    MEAN_AGGREGATOR,
    SUM_AGGREGATOR,
    L2_AGGREGATOR,
    L1_AGGREGATOR,
    STD_AGGREGATOR,
    MeanAggregator,
    SumAggregator,
    VectorNormAggregator,
    StdAggregator,
)
from .normalizers import (
    VARIANCE_NORMALIZER,
    STD_NORMALIZER,
    L2_NORM_NORMALIZER,
    L1_NORM_NORMALIZER,
    NAIVE_FORECAST_NORMALIZER,
    VarianceNormalizer,
    StandardDeviationNormalizer,
    VectorNormNormalizer,
    NaiveForecastNormalizer,
)


class MetricRegistry:
    """Thread-safe registry for metric classes."""

    def __init__(self):
        """Initialize the registry with built-in metrics."""
        self._metrics: dict[str, type[IMetric]] = {}
        self._lock = threading.RLock()
        self._register_builtin_metrics()

    def register(self, name: str, metric_class: type[IMetric]) -> None:
        """Register a metric class.

        Args:
            name: Unique metric name
            metric_class: Metric class implementing IMetric

        Raises:
            ValueError: If name already exists or metric_class is invalid
        """
        if not hasattr(metric_class, "compute"):
            raise ValueError(f"Metric class {metric_class} must implement compute method")

        with self._lock:
            if name in self._metrics:
                raise ValueError(f"Metric '{name}' is already registered")
            self._metrics[name] = metric_class

    def get(self, name: str) -> type[IMetric]:
        """Get registered metric class by name.

        Args:
            name: Metric name

        Returns:
            Metric class

        Raises:
            KeyError: If metric not found
        """
        with self._lock:
            if name not in self._metrics:
                raise KeyError(f"Metric '{name}' not found in registry")
            return self._metrics[name]

    def list_metrics(self) -> list[str]:
        """List all registered metric names."""
        with self._lock:
            return list(self._metrics.keys())

    def unregister(self, name: str) -> None:
        """Unregister a metric.

        Args:
            name: Metric name to remove

        Raises:
            KeyError: If metric not found
        """
        with self._lock:
            if name not in self._metrics:
                raise KeyError(f"Metric '{name}' not found in registry")
            del self._metrics[name]

    def _register_builtin_metrics(self) -> None:
        """Register built-in metric implementations."""
        builtin_metrics = {
            "mse": MeanSquaredErrorMetric,
            "mae": MeanAbsoluteErrorMetric,
            "rmse": RootMeanSquaredErrorMetric,
            "normalized_vector_norm_error": NormalizedVectorNormErrorMetric,
            "mse_over_var": MSEOverVarianceMetric,
            "temporal_derivative": TemporalDerivativeMetric,
        }

        for name, metric_class in builtin_metrics.items():
            self._metrics[name] = metric_class


class AggregatorRegistry:
    """Registry for aggregation strategies."""

    def __init__(self):
        """Initialize with built-in aggregators."""
        self._aggregators: dict[str, IAggregator] = {
            "mean": MEAN_AGGREGATOR,
            "sum": SUM_AGGREGATOR,
            "l2_norm": L2_AGGREGATOR,
            "l1_norm": L1_AGGREGATOR,
            "std": STD_AGGREGATOR,
        }
        self._lock = threading.RLock()

    def register(self, name: str, aggregator: IAggregator) -> None:
        """Register an aggregator."""
        with self._lock:
            self._aggregators[name] = aggregator

    def get(self, name: str) -> IAggregator:
        """Get aggregator by name."""
        with self._lock:
            if name not in self._aggregators:
                raise KeyError(f"Aggregator '{name}' not found")
            return self._aggregators[name]

    def list_aggregators(self) -> list[str]:
        """List all registered aggregator names."""
        with self._lock:
            return list(self._aggregators.keys())


class NormalizerRegistry:
    """Registry for normalization strategies."""

    def __init__(self):
        """Initialize with built-in normalizers."""
        self._normalizers: dict[str, INormalizer] = {
            "variance": VARIANCE_NORMALIZER,
            "std": STD_NORMALIZER,
            "l2_norm": L2_NORM_NORMALIZER,
            "l1_norm": L1_NORM_NORMALIZER,
            "naive_forecast": NAIVE_FORECAST_NORMALIZER,
        }
        self._lock = threading.RLock()

    def register(self, name: str, normalizer: INormalizer) -> None:
        """Register a normalizer."""
        with self._lock:
            self._normalizers[name] = normalizer

    def get(self, name: str) -> INormalizer:
        """Get normalizer by name."""
        with self._lock:
            if name not in self._normalizers:
                raise KeyError(f"Normalizer '{name}' not found")
            return self._normalizers[name]

    def list_normalizers(self) -> list[str]:
        """List all registered normalizer names."""
        with self._lock:
            return list(self._normalizers.keys())


class MetricFactory:
    """Factory for creating metrics with dependency injection."""

    def __init__(
        self,
        metric_registry: MetricRegistry | None = None,
        aggregator_registry: AggregatorRegistry | None = None,
        normalizer_registry: NormalizerRegistry | None = None,
    ):
        """Initialize factory with registries.

        Args:
            metric_registry: Registry for metric classes (default: global registry)
            aggregator_registry: Registry for aggregators (default: global registry)
            normalizer_registry: Registry for normalizers (default: global registry)
        """
        self._metric_registry = metric_registry or get_global_metric_registry()
        self._aggregator_registry = aggregator_registry or get_global_aggregator_registry()
        self._normalizer_registry = normalizer_registry or get_global_normalizer_registry()

    def create_metric(
        self,
        metric_type: str,
        aggregator: str | None = None,
        normalizer: str | None = None,
        **kwargs,
    ) -> IMetric:
        """Create metric instance with dependencies injected.

        Args:
            metric_type: Type of metric to create
            aggregator: Name of aggregation strategy (optional)
            normalizer: Name of normalization strategy (optional)
            **kwargs: Additional metric-specific parameters

        Returns:
            Configured metric instance

        Raises:
            KeyError: If metric type, aggregator, or normalizer not found
            ValueError: If parameters are invalid
        """
        # Get metric class
        metric_class = self._metric_registry.get(metric_type)

        # Get aggregator if specified
        aggregator_instance = None
        if aggregator is not None:
            aggregator_instance = self._aggregator_registry.get(aggregator)

        # Get normalizer if specified
        normalizer_instance = None
        if normalizer is not None:
            normalizer_instance = self._normalizer_registry.get(normalizer)

        # Create metric instance
        try:
            return metric_class(
                aggregator=aggregator_instance, normalizer=normalizer_instance, **kwargs
            )
        except TypeError as e:
            raise ValueError(f"Failed to create metric '{metric_type}': {e}") from e

    def create_normalized_vector_norm_error(
        self,
        vector_dim: int = -1,
        norm_ord: int = 2,
        aggregator: str = "mean",
        eps: float = 1e-8,
        **kwargs,
    ) -> NormalizedVectorNormErrorMetric:
        """Create normalized vector norm error metric with specific parameters.

        Args:
            vector_dim: Dimension along which vectors are defined
            norm_ord: Order of the norm (1, 2, etc.)
            aggregator: Name of aggregation strategy
            eps: Numerical stability parameter
            **kwargs: Additional parameters

        Returns:
            Configured NormalizedVectorNormErrorMetric instance
        """
        return self.create_metric(
            metric_type="normalized_vector_norm_error",
            aggregator=aggregator,
            vector_dim=vector_dim,
            norm_ord=norm_ord,
            eps=eps,
            **kwargs,
        )

    def create_custom_aggregator(self, aggregator_type: str, **params) -> IAggregator:
        """Create custom aggregator instances.

        Args:
            aggregator_type: Type of aggregator ("vector_norm", etc.)
            **params: Aggregator-specific parameters

        Returns:
            Configured aggregator instance
        """
        if aggregator_type == "vector_norm":
            ord_value = params.get("ord", 2)
            return VectorNormAggregator(ord=ord_value)
        elif aggregator_type == "mean":
            return MeanAggregator()
        elif aggregator_type == "sum":
            return SumAggregator()
        elif aggregator_type == "std":
            return StdAggregator()
        else:
            raise ValueError(f"Unknown aggregator type: {aggregator_type}")

    def create_custom_normalizer(self, normalizer_type: str, **params) -> INormalizer:
        """Create custom normalizer instances.

        Args:
            normalizer_type: Type of normalizer ("vector_norm", etc.)
            **params: Normalizer-specific parameters

        Returns:
            Configured normalizer instance
        """
        if normalizer_type == "vector_norm":
            ord_value = params.get("ord", 2)
            dim = params.get("dim", -1)
            return VectorNormNormalizer(ord=ord_value, dim=dim)
        elif normalizer_type == "variance":
            return VarianceNormalizer()
        elif normalizer_type == "std":
            return StandardDeviationNormalizer()
        elif normalizer_type == "naive_forecast":
            return NaiveForecastNormalizer()
        else:
            raise ValueError(f"Unknown normalizer type: {normalizer_type}")


# Global registries (singletons)
_global_metric_registry: MetricRegistry | None = None
_global_aggregator_registry: AggregatorRegistry | None = None
_global_normalizer_registry: NormalizerRegistry | None = None
_registry_lock = threading.Lock()


def get_global_metric_registry() -> MetricRegistry:
    """Get the global metric registry (singleton)."""
    global _global_metric_registry
    if _global_metric_registry is None:
        with _registry_lock:
            if _global_metric_registry is None:
                _global_metric_registry = MetricRegistry()
    return _global_metric_registry


def get_global_aggregator_registry() -> AggregatorRegistry:
    """Get the global aggregator registry (singleton)."""
    global _global_aggregator_registry
    if _global_aggregator_registry is None:
        with _registry_lock:
            if _global_aggregator_registry is None:
                _global_aggregator_registry = AggregatorRegistry()
    return _global_aggregator_registry


def get_global_normalizer_registry() -> NormalizerRegistry:
    """Get the global normalizer registry (singleton)."""
    global _global_normalizer_registry
    if _global_normalizer_registry is None:
        with _registry_lock:
            if _global_normalizer_registry is None:
                _global_normalizer_registry = NormalizerRegistry()
    return _global_normalizer_registry


def get_global_metric_factory() -> MetricFactory:
    """Get a metric factory using global registries."""
    return MetricFactory()
