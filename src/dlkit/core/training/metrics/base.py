"""Base metric classes implementing Template Method Pattern."""

from abc import ABC, abstractmethod
from typing import Any
import torch
from torch import Tensor

from .protocols import IMetric, IAggregator, INormalizer
from .aggregators import MEAN_AGGREGATOR


class BaseMetric(ABC):
    """Abstract base metric implementing Template Method Pattern.

    This class defines the common structure for all metrics while allowing
    subclasses to customize specific computation steps.
    """

    def __init__(
        self,
        name: str,
        aggregator: IAggregator | None = None,
        normalizer: INormalizer | None = None,
        **kwargs,
    ):
        """Initialize base metric.

        Args:
            name: Metric name
            aggregator: Strategy for aggregating values (default: mean)
            normalizer: Strategy for normalization (default: variance)
            **kwargs: Additional metric-specific parameters
        """
        self._name = name
        self._aggregator = aggregator or MEAN_AGGREGATOR
        self._normalizer = normalizer  # No default normalizer - let subclasses decide
        self._params = kwargs

    @property
    def name(self) -> str:
        """Return metric name."""
        return self._name

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metric metadata."""
        return {
            "name": self._name,
            "aggregator": self._aggregator.name,
            "normalizer": self._normalizer.name if self._normalizer else None,
            "parameters": self._params,
        }

    def compute(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Template method for metric computation.

        This method defines the algorithm structure:
        1. Validate inputs
        2. Compute raw error
        3. Apply normalization
        4. Apply aggregation
        5. Post-process result

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional parameters

        Returns:
            Computed metric value
        """
        # Step 1: Validate inputs
        self._validate_inputs(predictions, targets)

        # Step 2: Compute raw error (implemented by subclasses)
        raw_error = self._compute_raw_error(predictions, targets, **kwargs)

        # Step 3: Apply normalization if needed
        if self._should_normalize():
            normalized_error = self._apply_normalization(raw_error, targets)
        else:
            normalized_error = raw_error

        # Step 4: Apply aggregation
        aggregated_result = self._apply_aggregation(normalized_error)

        # Step 5: Post-process result (hook for subclasses)
        return self._post_process(aggregated_result)

    def __call__(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Make metrics callable like functions.

        This allows metrics to be used directly as callables:

        Examples:
            >>> metric = MeanSquaredErrorMetric()
            >>> error = metric(predictions, targets)  # Function-like usage
            >>> error = metric.compute(predictions, targets)  # Method usage

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional parameters

        Returns:
            Computed metric value
        """
        return self.compute(predictions, targets, **kwargs)

    def _validate_inputs(self, predictions: Tensor, targets: Tensor) -> None:
        """Validate input tensors.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(predictions, Tensor) or not isinstance(targets, Tensor):
            raise TypeError("Both predictions and targets must be torch.Tensor")

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} != targets {targets.shape}"
            )

    @abstractmethod
    def _compute_raw_error(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute raw error values (to be implemented by subclasses).

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional parameters

        Returns:
            Raw error tensor
        """

    def _should_normalize(self) -> bool:
        """Determine if normalization should be applied."""
        return self._normalizer is not None

    def _apply_normalization(self, error: Tensor, targets: Tensor) -> Tensor:
        """Apply normalization using configured strategy."""
        eps = self._params.get("eps", 1e-8)
        return self._normalizer.normalize(error, targets, eps=eps)

    def _apply_aggregation(self, values: Tensor) -> Tensor:
        """Apply aggregation using configured strategy."""
        dim = self._params.get("dim", None)
        return self._aggregator.aggregate(values, dim=dim)

    def _post_process(self, result: Tensor) -> Tensor:
        """Post-process result (hook for subclasses)."""
        return result


class CompositeMetric:
    """Composite metric for combining multiple metrics."""

    def __init__(self, name: str, metrics: list[IMetric], weights: Tensor | None = None):
        """Initialize composite metric.

        Args:
            name: Composite metric name
            metrics: List of metrics to combine
            weights: Optional weights for weighted combination
        """
        self._name = name
        self._metrics = metrics
        self._weights = weights

        if weights is not None and len(weights) != len(metrics):
            raise ValueError("Number of weights must match number of metrics")

    @property
    def name(self) -> str:
        """Return composite metric name."""
        return self._name

    @property
    def metadata(self) -> dict[str, Any]:
        """Return composite metric metadata."""
        return {
            "name": self._name,
            "component_metrics": [metric.name for metric in self._metrics],
            "weights": self._weights.tolist() if self._weights is not None else None,
        }

    def compute(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute composite metric as weighted combination of components."""
        results = []
        for metric in self._metrics:
            result = metric.compute(predictions, targets, **kwargs)
            results.append(result)

        # Stack results and apply weights
        stacked_results = torch.stack(results)

        if self._weights is not None:
            # Ensure weights have correct shape for broadcasting
            weights = self._weights.view(-1, *([1] * (stacked_results.dim() - 1)))
            weighted_results = stacked_results * weights
            return torch.sum(weighted_results, dim=0)
        else:
            return torch.mean(stacked_results, dim=0)

    def __call__(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Make composite metrics callable like functions.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional parameters

        Returns:
            Computed composite metric value
        """
        return self.compute(predictions, targets, **kwargs)


class MetricDecorator(ABC):
    """Abstract decorator for adding functionality to metrics."""

    def __init__(self, metric: IMetric):
        """Initialize decorator with wrapped metric.

        Args:
            metric: Metric to decorate
        """
        self._wrapped_metric = metric

    @property
    def name(self) -> str:
        """Return decorated metric name."""
        return f"{self.__class__.__name__}({self._wrapped_metric.name})"

    @property
    def metadata(self) -> dict[str, Any]:
        """Return decorated metric metadata."""
        base_metadata = self._wrapped_metric.metadata
        base_metadata["decorator"] = self.__class__.__name__
        return base_metadata

    @abstractmethod
    def compute(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute decorated metric."""

    def __call__(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Make decorated metrics callable like functions.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional parameters

        Returns:
            Computed decorated metric value
        """
        return self.compute(predictions, targets, **kwargs)
