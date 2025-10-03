"""Tests for callable metrics functionality."""

import pytest
import torch
from torch import Tensor

from dlkit.core.training.metrics import (
    create_metric,
    create_normalized_vector_norm_error,
    create_composite_metric,
    MeanSquaredErrorMetric,
    MeanAbsoluteErrorMetric,
)


class TestCallableMetrics:
    """Tests for metrics callable like functions."""

    def test_basic_metric_callable(self, sample_predictions, sample_targets):
        """Test that basic metrics can be called like functions."""
        mse = create_metric("mse")
        mae = create_metric("mae")

        # Test method vs callable syntax
        mse_method_result = mse.compute(sample_predictions, sample_targets)
        mse_callable_result = mse(sample_predictions, sample_targets)

        mae_method_result = mae.compute(sample_predictions, sample_targets)
        mae_callable_result = mae(sample_predictions, sample_targets)

        # Results should be identical
        assert torch.allclose(mse_method_result, mse_callable_result)
        assert torch.allclose(mae_method_result, mae_callable_result)

    def test_normalized_vector_norm_callable(self, sample_2d_vectors):
        """Test that normalized vector norm error can be called like a function."""
        predictions, targets = sample_2d_vectors
        metric = create_normalized_vector_norm_error(vector_dim=-1, norm_ord=2)

        # Test method vs callable syntax
        method_result = metric.compute(predictions, targets)
        callable_result = metric(predictions, targets)

        # Results should be identical
        assert torch.allclose(method_result, callable_result)

    def test_composite_metric_callable(self, sample_predictions, sample_targets):
        """Test that composite metrics can be called like functions."""
        mse = create_metric("mse")
        mae = create_metric("mae")
        composite = create_composite_metric("mse_mae", [mse, mae], weights=[0.7, 0.3])

        # Test method vs callable syntax
        method_result = composite.compute(sample_predictions, sample_targets)
        callable_result = composite(sample_predictions, sample_targets)

        # Results should be identical
        assert torch.allclose(method_result, callable_result)

    def test_direct_class_callable(self, sample_predictions, sample_targets):
        """Test that direct class instantiation supports callable syntax."""
        mse = MeanSquaredErrorMetric()
        mae = MeanAbsoluteErrorMetric()

        # Test method vs callable syntax
        mse_method = mse.compute(sample_predictions, sample_targets)
        mse_callable = mse(sample_predictions, sample_targets)

        mae_method = mae.compute(sample_predictions, sample_targets)
        mae_callable = mae(sample_predictions, sample_targets)

        # Results should be identical
        assert torch.allclose(mse_method, mse_callable)
        assert torch.allclose(mae_method, mae_callable)

    def test_callable_with_kwargs(self, sample_predictions, sample_targets):
        """Test that callable syntax supports keyword arguments."""
        # Create a metric that accepts additional parameters
        mse = create_metric("mse", eps=1e-10)

        # Test that kwargs work with callable syntax
        result_method = mse.compute(sample_predictions, sample_targets)
        result_callable = mse(sample_predictions, sample_targets)

        # Results should be identical
        assert torch.allclose(result_method, result_callable)

    def test_functional_programming_style(self, sample_predictions, sample_targets):
        """Test metrics in functional programming patterns."""
        metrics = [
            create_metric("mse"),
            create_metric("mae"),
            create_metric("rmse"),
        ]

        # Test list comprehension with callable syntax
        results = [metric(sample_predictions, sample_targets) for metric in metrics]

        # All results should be tensors
        assert all(isinstance(result, Tensor) for result in results)
        assert all(torch.isfinite(result).all() for result in results)

    def test_higher_order_functions(self, sample_predictions, sample_targets):
        """Test metrics with higher-order functions."""

        def apply_metric(metric_fn, pred, target):
            """Higher-order function that applies a metric."""
            return metric_fn(pred, target)

        mse = create_metric("mse")
        mae = create_metric("mae")

        # Test using metrics as function arguments
        mse_result = apply_metric(mse, sample_predictions, sample_targets)
        mae_result = apply_metric(mae, sample_predictions, sample_targets)

        # Results should be valid tensors
        assert torch.isfinite(mse_result).all()
        assert torch.isfinite(mae_result).all()

    def test_callable_preserves_protocol_compliance(self, sample_predictions, sample_targets):
        """Test that callable syntax maintains protocol compliance."""
        from dlkit.core.training.metrics import IMetric

        def process_metric(metric: IMetric, pred: Tensor, target: Tensor) -> Tensor:
            """Function that expects IMetric protocol."""
            # Should work with both syntaxes
            method_result = metric.compute(pred, target)
            callable_result = metric(pred, target)

            # Verify they're the same
            assert torch.allclose(method_result, callable_result)
            return callable_result

        mse = create_metric("mse")
        result = process_metric(mse, sample_predictions, sample_targets)

        assert torch.isfinite(result).all()

    def test_callable_error_handling(self):
        """Test that callable syntax preserves error handling."""
        mse = create_metric("mse")

        # Test that validation errors are preserved
        with pytest.raises(ValueError, match="Shape mismatch"):
            mse(torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0], [2.0]]))

        with pytest.raises(TypeError, match="must be torch.Tensor"):
            mse([1.0, 2.0], torch.tensor([1.0, 2.0]))

    def test_callable_with_map_function(self, sample_predictions, sample_targets):
        """Test metrics work with map() function."""
        mse = create_metric("mse")
        mae = create_metric("mae")

        # Create multiple prediction/target pairs
        pred_batches = [sample_predictions, sample_predictions * 0.5]
        target_batches = [sample_targets, sample_targets * 0.5]

        # Test with map function
        mse_results = list(map(mse, pred_batches, target_batches))
        mae_results = list(map(mae, pred_batches, target_batches))

        # All results should be valid tensors
        assert len(mse_results) == 2
        assert len(mae_results) == 2
        assert all(torch.isfinite(result).all() for result in mse_results)
        assert all(torch.isfinite(result).all() for result in mae_results)

    def test_callable_preserves_metadata(self, sample_predictions, sample_targets):
        """Test that callable metrics preserve metadata access."""
        mse = create_metric("mse")

        # Use callable syntax
        result = mse(sample_predictions, sample_targets)

        # Metadata should still be accessible
        assert mse.name == "mse"
        assert "name" in mse.metadata
        assert "aggregator" in mse.metadata
        assert torch.isfinite(result).all()

    @pytest.fixture
    def sample_2d_vectors(self):
        """Sample 2D vector dataflow for testing."""
        predictions = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        targets = torch.tensor([[1.0, 1.0], [2.0, 0.0], [0.0, 1.0]])
        return predictions, targets
