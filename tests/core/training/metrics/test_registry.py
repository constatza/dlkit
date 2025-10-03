"""Tests for metric registry and factory."""

import pytest
import torch

from dlkit.core.training.metrics import (
    MetricRegistry,
    AggregatorRegistry,
    NormalizerRegistry,
    MetricFactory,
    get_global_metric_registry,
    get_global_aggregator_registry,
    get_global_normalizer_registry,
    get_global_metric_factory,
    MeanSquaredErrorMetric,
    NormalizedVectorNormErrorMetric,
    BaseMetric,
)


class MockMetric(BaseMetric):
    """Mock metric for testing registration."""

    def __init__(self, **kwargs):
        super().__init__(name="mock", **kwargs)

    def _compute_raw_error(self, predictions, targets, **kwargs):
        return torch.abs(predictions - targets)


class TestMetricRegistry:
    """Tests for metric registry."""

    def test_register_and_get_metric(self):
        """Test registering and retrieving metrics."""
        registry = MetricRegistry()

        # Register a custom metric
        registry.register("custom_metric", MockMetric)

        # Retrieve the metric
        retrieved_class = registry.get("custom_metric")
        assert retrieved_class == MockMetric

    def test_register_duplicate_name_raises_error(self):
        """Test that registering duplicate names raises ValueError."""
        registry = MetricRegistry()
        registry.register("test_metric", MockMetric)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("test_metric", MeanSquaredErrorMetric)

    def test_get_nonexistent_metric_raises_error(self):
        """Test that getting non-existent metric raises KeyError."""
        registry = MetricRegistry()

        with pytest.raises(KeyError, match="not found in registry"):
            registry.get("nonexistent_metric")

    def test_list_metrics(self):
        """Test listing registered metrics."""
        registry = MetricRegistry()

        # Should contain built-in metrics
        metrics = registry.list_metrics()
        assert "mse" in metrics
        assert "mae" in metrics
        assert "normalized_vector_norm_error" in metrics

    def test_unregister_metric(self):
        """Test unregistering metrics."""
        registry = MetricRegistry()
        registry.register("temp_metric", MockMetric)

        # Verify it's registered
        assert "temp_metric" in registry.list_metrics()

        # Unregister it
        registry.unregister("temp_metric")
        assert "temp_metric" not in registry.list_metrics()

    def test_unregister_nonexistent_raises_error(self):
        """Test that unregistering non-existent metric raises KeyError."""
        registry = MetricRegistry()

        with pytest.raises(KeyError, match="not found in registry"):
            registry.unregister("nonexistent_metric")

    def test_builtin_metrics_registered(self):
        """Test that built-in metrics are automatically registered."""
        registry = MetricRegistry()

        # Check that key built-in metrics are present
        builtin_metrics = ["mse", "mae", "rmse", "normalized_vector_norm_error"]
        registered_metrics = registry.list_metrics()

        for metric_name in builtin_metrics:
            assert metric_name in registered_metrics

    def test_register_invalid_metric_raises_error(self):
        """Test that registering invalid metric class raises ValueError."""
        registry = MetricRegistry()

        class InvalidMetric:
            pass  # Doesn't implement compute method

        with pytest.raises(ValueError, match="must implement compute method"):
            registry.register("invalid", InvalidMetric)


class TestAggregatorRegistry:
    """Tests for aggregator registry."""

    def test_builtin_aggregators(self):
        """Test that built-in aggregators are available."""
        registry = AggregatorRegistry()

        builtin_aggregators = ["mean", "sum", "l2_norm", "l1_norm", "std"]
        available_aggregators = registry.list_aggregators()

        for aggregator_name in builtin_aggregators:
            assert aggregator_name in available_aggregators

    def test_get_aggregator(self):
        """Test getting aggregator by name."""
        registry = AggregatorRegistry()

        mean_aggregator = registry.get("mean")
        assert mean_aggregator.name == "mean"

    def test_get_nonexistent_aggregator_raises_error(self):
        """Test that getting non-existent aggregator raises KeyError."""
        registry = AggregatorRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent_aggregator")


class TestNormalizerRegistry:
    """Tests for normalizer registry."""

    def test_builtin_normalizers(self):
        """Test that built-in normalizers are available."""
        registry = NormalizerRegistry()

        builtin_normalizers = ["variance", "std", "l2_norm", "l1_norm", "naive_forecast"]
        available_normalizers = registry.list_normalizers()

        for normalizer_name in builtin_normalizers:
            assert normalizer_name in available_normalizers

    def test_get_normalizer(self):
        """Test getting normalizer by name."""
        registry = NormalizerRegistry()

        variance_normalizer = registry.get("variance")
        assert variance_normalizer.name == "variance"

    def test_get_nonexistent_normalizer_raises_error(self):
        """Test that getting non-existent normalizer raises KeyError."""
        registry = NormalizerRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent_normalizer")


class TestMetricFactory:
    """Tests for metric factory."""

    def test_create_basic_metric(self):
        """Test creating basic metric without custom components."""
        factory = MetricFactory()

        metric = factory.create_metric("mse")
        assert isinstance(metric, MeanSquaredErrorMetric)
        assert metric.name == "mse"

    def test_create_metric_with_aggregator(self):
        """Test creating metric with custom aggregator."""
        factory = MetricFactory()

        metric = factory.create_metric("mse", aggregator="sum")
        assert isinstance(metric, MeanSquaredErrorMetric)
        assert metric._aggregator.name == "sum"

    def test_create_metric_with_normalizer(self):
        """Test creating metric with custom normalizer."""
        factory = MetricFactory()

        metric = factory.create_metric("mse", normalizer="std")
        assert isinstance(metric, MeanSquaredErrorMetric)
        assert metric._normalizer.name == "std"

    def test_create_metric_with_both_aggregator_and_normalizer(self):
        """Test creating metric with both custom aggregator and normalizer."""
        factory = MetricFactory()

        metric = factory.create_metric("mse", aggregator="l2_norm", normalizer="variance")
        assert isinstance(metric, MeanSquaredErrorMetric)
        assert metric._aggregator.name == "vector_norm_ord_2"
        assert metric._normalizer.name == "variance"

    def test_create_metric_with_kwargs(self):
        """Test creating metric with additional parameters."""
        factory = MetricFactory()

        metric = factory.create_metric("mse", eps=1e-10, dim=0)
        assert metric._params["eps"] == 1e-10
        assert metric._params["dim"] == 0

    @pytest.fixture
    def sample_2d_vectors(self):
        """Sample 2D vector dataflow for testing."""
        predictions = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        targets = torch.tensor([[1.0, 1.0], [2.0, 0.0], [0.0, 1.0]])
        return predictions, targets

    def test_create_normalized_vector_norm_error(self, sample_2d_vectors):
        """Test creating normalized vector norm error metric."""
        factory = MetricFactory()
        predictions, targets = sample_2d_vectors

        metric = factory.create_normalized_vector_norm_error(
            vector_dim=-1, norm_ord=1, aggregator="sum", eps=1e-6
        )

        assert isinstance(metric, NormalizedVectorNormErrorMetric)
        assert metric._params["norm_ord"] == 1
        assert metric._params["eps"] == 1e-6
        assert metric._aggregator.name == "sum"

        # Test that it computes correctly
        result = metric.compute(predictions, targets)
        assert torch.isfinite(result).all()

    def test_create_metric_nonexistent_type_raises_error(self):
        """Test that creating non-existent metric type raises KeyError."""
        factory = MetricFactory()

        with pytest.raises(KeyError, match="not found in registry"):
            factory.create_metric("nonexistent_metric")

    def test_create_metric_nonexistent_aggregator_raises_error(self):
        """Test that using non-existent aggregator raises KeyError."""
        factory = MetricFactory()

        with pytest.raises(KeyError, match="not found"):
            factory.create_metric("mse", aggregator="nonexistent_aggregator")

    def test_create_metric_nonexistent_normalizer_raises_error(self):
        """Test that using non-existent normalizer raises KeyError."""
        factory = MetricFactory()

        with pytest.raises(KeyError, match="not found"):
            factory.create_metric("mse", normalizer="nonexistent_normalizer")

    def test_create_metric_accepts_custom_kwargs(self):
        """Test that custom kwargs are accepted and stored in parameters."""
        factory = MetricFactory()

        # Custom parameters should be accepted and stored
        metric = factory.create_metric("mse", custom_param="custom_value", eps=1e-10)
        assert metric._params["custom_param"] == "custom_value"
        assert metric._params["eps"] == 1e-10

    def test_create_custom_aggregator(self):
        """Test creating custom aggregator instances."""
        factory = MetricFactory()

        # Test vector norm aggregator
        aggregator = factory.create_custom_aggregator("vector_norm", ord=1)
        assert aggregator.name == "vector_norm_ord_1"

        # Test other aggregators
        mean_agg = factory.create_custom_aggregator("mean")
        assert mean_agg.name == "mean"

    def test_create_custom_normalizer(self):
        """Test creating custom normalizer instances."""
        factory = MetricFactory()

        # Test vector norm normalizer
        normalizer = factory.create_custom_normalizer("vector_norm", ord=1, dim=0)
        assert normalizer.name == "vector_norm_ord_1_dim_0"

        # Test other normalizers
        var_norm = factory.create_custom_normalizer("variance")
        assert var_norm.name == "variance"

    def test_create_custom_aggregator_unknown_type_raises_error(self):
        """Test that unknown aggregator type raises ValueError."""
        factory = MetricFactory()

        with pytest.raises(ValueError, match="Unknown aggregator type"):
            factory.create_custom_aggregator("unknown_type")

    def test_create_custom_normalizer_unknown_type_raises_error(self):
        """Test that unknown normalizer type raises ValueError."""
        factory = MetricFactory()

        with pytest.raises(ValueError, match="Unknown normalizer type"):
            factory.create_custom_normalizer("unknown_type")


class TestGlobalRegistries:
    """Tests for global registry functions."""

    def test_global_metric_registry_singleton(self):
        """Test that global metric registry is a singleton."""
        registry1 = get_global_metric_registry()
        registry2 = get_global_metric_registry()

        assert registry1 is registry2

    def test_global_aggregator_registry_singleton(self):
        """Test that global aggregator registry is a singleton."""
        registry1 = get_global_aggregator_registry()
        registry2 = get_global_aggregator_registry()

        assert registry1 is registry2

    def test_global_normalizer_registry_singleton(self):
        """Test that global normalizer registry is a singleton."""
        registry1 = get_global_normalizer_registry()
        registry2 = get_global_normalizer_registry()

        assert registry1 is registry2

    def test_global_metric_factory(self):
        """Test global metric factory."""
        factory = get_global_metric_factory()

        # Should be able to create metrics using global registries
        metric = factory.create_metric("mse")
        assert isinstance(metric, MeanSquaredErrorMetric)
