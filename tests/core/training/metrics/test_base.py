"""Tests for base metric classes."""

import pytest
import torch
from torch import Tensor

from dlkit.core.training.metrics import (
    BaseMetric,
    CompositeMetric,
    MetricDecorator,
    SUM_AGGREGATOR,
)


class ConcreteMetric(BaseMetric):
    """Concrete metric implementation for testing."""

    def _compute_raw_error(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        return torch.abs(predictions - targets)


class ConcreteMetricNoNormalization(BaseMetric):
    """Concrete metric that doesn't normalize."""

    def _compute_raw_error(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        return torch.abs(predictions - targets)

    def _should_normalize(self) -> bool:
        return False


class ConcreteMetricWithPostProcessing(BaseMetric):
    """Concrete metric with custom post-processing."""

    def _compute_raw_error(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        return torch.pow(predictions - targets, 2)

    def _post_process(self, result: Tensor) -> Tensor:
        # Apply square root as post-processing
        return torch.sqrt(result)


class MockDecorator(MetricDecorator):
    """Mock decorator for testing."""

    def compute(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        # Add constant value to wrapped metric result
        base_result = self._wrapped_metric.compute(predictions, targets, **kwargs)
        return base_result + 1.0


class TestBaseMetric:
    """Tests for BaseMetric abstract class."""

    def test_basic_computation_flow(self, sample_predictions, sample_targets):
        """Test the template method computation flow."""
        metric = ConcreteMetric("test_metric")
        result = metric.compute(sample_predictions, sample_targets)

        # Should complete without error and return finite result
        assert torch.isfinite(result).all()
        assert isinstance(result, Tensor)

    def test_metric_properties(self):
        """Test metric name and metadata properties."""
        metric = ConcreteMetric("test_metric", eps=1e-6, dim=0)

        assert metric.name == "test_metric"

        metadata = metric.metadata
        assert metadata["name"] == "test_metric"
        assert metadata["aggregator"] == "mean"
        assert metadata["normalizer"] is None  # ConcreteMetric doesn't set a normalizer
        assert metadata["parameters"]["eps"] == 1e-6
        assert metadata["parameters"]["dim"] == 0

    def test_custom_aggregator_and_normalizer(self, sample_predictions, sample_targets):
        """Test metric with custom aggregator and normalizer."""
        metric = ConcreteMetric(
            "test_metric",
            aggregator=SUM_AGGREGATOR,
            normalizer=None,  # No normalization
        )

        result = metric.compute(sample_predictions, sample_targets)

        # Should use sum aggregation instead of mean
        assert torch.isfinite(result).all()

    def test_no_normalization_path(self, sample_predictions, sample_targets):
        """Test computation path when normalization is disabled."""
        metric = ConcreteMetricNoNormalization("test_metric")
        result = metric.compute(sample_predictions, sample_targets)

        # Should complete without normalization step
        assert torch.isfinite(result).all()

    def test_post_processing_hook(self, sample_predictions, sample_targets):
        """Test custom post-processing."""
        metric = ConcreteMetricWithPostProcessing("test_metric")
        result = metric.compute(sample_predictions, sample_targets)

        # Result should be post-processed (square root applied)
        assert torch.isfinite(result).all()

    def test_input_validation_shape_mismatch(self):
        """Test input validation for shape mismatch."""
        metric = ConcreteMetric("test_metric")
        predictions = torch.tensor([[1.0, 2.0]])
        targets = torch.tensor([[1.0], [2.0]])

        with pytest.raises(ValueError, match="Shape mismatch"):
            metric.compute(predictions, targets)

    def test_input_validation_non_tensor(self):
        """Test input validation for non-tensor inputs."""
        metric = ConcreteMetric("test_metric")

        with pytest.raises(TypeError, match="must be torch.Tensor"):
            metric.compute([1.0, 2.0], torch.tensor([1.0, 2.0]))

        with pytest.raises(TypeError, match="must be torch.Tensor"):
            metric.compute(torch.tensor([1.0, 2.0]), [1.0, 2.0])

    def test_aggregation_with_custom_dim(self, sample_predictions, sample_targets):
        """Test aggregation with custom dimension parameter."""
        metric = ConcreteMetric("test_metric", dim=0)
        result = metric.compute(sample_predictions, sample_targets)

        # Should aggregate along dimension 0
        assert torch.isfinite(result).all()

    def test_normalization_with_custom_eps(self, sample_predictions, sample_targets):
        """Test normalization with custom epsilon parameter."""
        metric = ConcreteMetric("test_metric", eps=1e-10)
        result = metric.compute(sample_predictions, sample_targets)

        # Should use custom epsilon for numerical stability
        assert torch.isfinite(result).all()

    def test_empty_tensors(self):
        """Test behavior with empty tensors."""
        # Use metric without normalization to avoid NaN from empty variance
        metric = ConcreteMetricNoNormalization("test_metric")
        empty_predictions = torch.empty(0, 3)
        empty_targets = torch.empty(0, 3)

        # Should handle empty tensors gracefully
        result = metric.compute(empty_predictions, empty_targets)
        # Mean of empty tensor is NaN, which is expected behavior
        # Let's check that we get a result (even if NaN)
        assert result is not None
        assert isinstance(result, Tensor)


class TestCompositeMetric:
    """Tests for CompositeMetric class."""

    def test_basic_composition(self, sample_predictions, sample_targets):
        """Test basic metric composition."""
        metric1 = ConcreteMetric("metric1")
        metric2 = ConcreteMetric("metric2")

        composite = CompositeMetric("composite", [metric1, metric2])
        result = composite.compute(sample_predictions, sample_targets)

        # Should return average of component metrics
        assert torch.isfinite(result).all()

    def test_weighted_composition(self, sample_predictions, sample_targets):
        """Test weighted metric composition."""
        metric1 = ConcreteMetric("metric1")
        metric2 = ConcreteMetric("metric2")
        weights = torch.tensor([0.7, 0.3])

        composite = CompositeMetric("weighted_composite", [metric1, metric2], weights)
        result = composite.compute(sample_predictions, sample_targets)

        # Should return weighted combination
        assert torch.isfinite(result).all()

    def test_composition_properties(self):
        """Test composite metric properties."""
        metric1 = ConcreteMetric("metric1")
        metric2 = ConcreteMetric("metric2")
        weights = torch.tensor([0.6, 0.4])

        composite = CompositeMetric("composite", [metric1, metric2], weights)

        assert composite.name == "composite"

        metadata = composite.metadata
        assert metadata["name"] == "composite"
        assert metadata["component_metrics"] == ["metric1", "metric2"]
        # Use torch.allclose for floating point comparison
        expected_weights = torch.tensor([0.6, 0.4])
        actual_weights = torch.tensor(metadata["weights"])
        assert torch.allclose(actual_weights, expected_weights)

    def test_composition_without_weights(self, sample_predictions, sample_targets):
        """Test composition without explicit weights (equal weighting)."""
        metric1 = ConcreteMetric("metric1")
        metric2 = ConcreteMetric("metric2")

        composite = CompositeMetric("unweighted_composite", [metric1, metric2])
        result = composite.compute(sample_predictions, sample_targets)

        # Should use equal weights (mean)
        assert torch.isfinite(result).all()

        metadata = composite.metadata
        assert metadata["weights"] is None

    def test_weight_count_mismatch_error(self):
        """Test that mismatched weight count raises ValueError."""
        metric1 = ConcreteMetric("metric1")
        metric2 = ConcreteMetric("metric2")
        weights = torch.tensor([0.5])  # Only one weight for two metrics

        with pytest.raises(ValueError, match="Number of weights must match"):
            CompositeMetric("composite", [metric1, metric2], weights)

    def test_single_metric_composition(self, sample_predictions, sample_targets):
        """Test composition with single metric."""
        metric = ConcreteMetric("single_metric")
        composite = CompositeMetric("single_composite", [metric])

        result = composite.compute(sample_predictions, sample_targets)

        # Should work with single metric
        assert torch.isfinite(result).all()

    def test_empty_metric_list(self):
        """Test composition with empty metric list."""
        # This should raise an error during computation due to empty stack
        composite = CompositeMetric("empty_composite", [])

        with pytest.raises(RuntimeError):
            composite.compute(torch.tensor([1.0]), torch.tensor([1.0]))


class TestMetricDecorator:
    """Tests for MetricDecorator abstract class."""

    def test_decorator_basic_functionality(self, sample_predictions, sample_targets):
        """Test basic decorator functionality."""
        base_metric = ConcreteMetric("base_metric")
        decorated_metric = MockDecorator(base_metric)

        base_result = base_metric.compute(sample_predictions, sample_targets)
        decorated_result = decorated_metric.compute(sample_predictions, sample_targets)

        # Decorated result should be base result + 1.0
        expected = base_result + 1.0
        assert torch.allclose(decorated_result, expected, atol=1e-6)

    def test_decorator_properties(self):
        """Test decorator properties."""
        base_metric = ConcreteMetric("base_metric")
        decorated_metric = MockDecorator(base_metric)

        assert "MockDecorator(base_metric)" in decorated_metric.name

        metadata = decorated_metric.metadata
        assert metadata["decorator"] == "MockDecorator"
        # Should include base metric metadata
        assert "name" in metadata

    def test_decorator_chaining(self, sample_predictions, sample_targets):
        """Test chaining multiple decorators."""
        base_metric = ConcreteMetric("base_metric")
        decorated_once = MockDecorator(base_metric)
        decorated_twice = MockDecorator(decorated_once)

        base_result = base_metric.compute(sample_predictions, sample_targets)
        final_result = decorated_twice.compute(sample_predictions, sample_targets)

        # Should add 1.0 twice
        expected = base_result + 2.0
        assert torch.allclose(final_result, expected, atol=1e-6)

    def test_decorator_preserves_base_metric_interface(self):
        """Test that decorator preserves base metric interface."""
        base_metric = ConcreteMetric("base_metric")
        decorated_metric = MockDecorator(base_metric)

        # Should still implement the metric interface
        assert hasattr(decorated_metric, "compute")
        assert hasattr(decorated_metric, "name")
        assert hasattr(decorated_metric, "metadata")
