"""Protocol definitions for metrics system following SOLID principles."""

from typing import Protocol, Any
from torch import Tensor


class IMetric(Protocol):
    """Interface for all metrics following Interface Segregation Principle."""

    def compute(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute the metric value.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional metric-specific parameters

        Returns:
            Computed metric value as tensor
        """
        ...

    def __call__(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Make metrics callable like functions.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional metric-specific parameters

        Returns:
            Computed metric value as tensor
        """
        ...

    @property
    def name(self) -> str:
        """Return the metric name."""
        ...

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metric metadata (dimensions, parameters, etc.)."""
        ...


class IAggregator(Protocol):
    """Interface for metric aggregation strategies."""

    def aggregate(self, values: Tensor, dim: int | None = None) -> Tensor:
        """Aggregate tensor values along specified dimension.

        Args:
            values: Tensor values to aggregate
            dim: Dimension to aggregate along (None for all dimensions)

        Returns:
            Aggregated tensor
        """
        ...

    @property
    def name(self) -> str:
        """Return aggregator name."""
        ...


class INormalizer(Protocol):
    """Interface for normalization strategies."""

    def normalize(self, values: Tensor, reference: Tensor, eps: float = 1e-8) -> Tensor:
        """Normalize values using reference tensor.

        Args:
            values: Values to normalize
            reference: Reference tensor for normalization
            eps: Small epsilon for numerical stability

        Returns:
            Normalized tensor
        """
        ...

    @property
    def name(self) -> str:
        """Return normalizer name."""
        ...


class IMetricComposer(Protocol):
    """Interface for composing multiple metrics."""

    def compose(self, *metrics: IMetric, weights: Tensor | None = None) -> IMetric:
        """Compose multiple metrics into a single metric.

        Args:
            *metrics: Metrics to compose
            weights: Optional weights for weighted composition

        Returns:
            Composed metric implementing IMetric
        """
        ...


class IMetricRegistry(Protocol):
    """Interface for metric registration and discovery."""

    def register(self, name: str, metric_class: type[IMetric]) -> None:
        """Register a metric class.

        Args:
            name: Unique metric name
            metric_class: Metric class implementing IMetric
        """
        ...

    def get(self, name: str) -> type[IMetric]:
        """Get registered metric class by name.

        Args:
            name: Metric name

        Returns:
            Metric class

        Raises:
            KeyError: If metric not found
        """
        ...

    def list_metrics(self) -> list[str]:
        """List all registered metric names."""
        ...


class IMetricFactory(Protocol):
    """Interface for metric instantiation following Dependency Inversion."""

    def create_metric(
        self,
        metric_type: str,
        aggregator: IAggregator | None = None,
        normalizer: INormalizer | None = None,
        **kwargs,
    ) -> IMetric:
        """Create metric instance with dependencies injected.

        Args:
            metric_type: Type of metric to create
            aggregator: Optional aggregation strategy
            normalizer: Optional normalization strategy
            **kwargs: Additional metric-specific parameters

        Returns:
            Configured metric instance
        """
        ...
