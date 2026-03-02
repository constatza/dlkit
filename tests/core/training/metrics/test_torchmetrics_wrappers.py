"""Comprehensive tests for torchmetrics wrappers.

Tests focus on:
    - State accumulation across batches
    - MetricCollection compatibility
    - Consistency with functional implementations
    - Distributed reduction behavior
    - Reset functionality
"""

import pytest
import torch
from torch import Tensor
from torchmetrics import MetricCollection

from dlkit.core.training.metrics.torchmetrics_wrappers import (
    NormalizedVectorNormError,
    TemporalDerivativeError,
)
from dlkit.core.training.metrics.functional import (
    normalized_vector_norm_error,
    temporal_derivative_error,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def simple_2d_vectors():
    """Simple 2D vectors for basic testing."""
    preds = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    target = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    return preds, target


@pytest.fixture
def temporal_sequence_3d():
    """Temporal sequence with shape (B=2, T=4, D=2)."""
    preds = torch.tensor([
        [[0.0, 0.0], [1.0, 0.5], [2.5, 1.5], [4.0, 2.0]],
        [[0.0, 1.0], [1.5, 2.0], [3.0, 3.5], [4.5, 5.0]],
    ])
    target = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
    ])
    return preds, target


# ============================================================================
# 1. NORMALIZED VECTOR NORM ERROR TESTS
# ============================================================================


class TestNormalizedVectorNormError:
    """Test NormalizedVectorNormError wrapper."""

    def test_single_batch(self, simple_2d_vectors):
        """Test with single batch."""
        preds, target = simple_2d_vectors
        metric = NormalizedVectorNormError(vector_dim=-1, norm_ord=2)

        metric.update(preds, target)
        result = metric.compute()

        assert result.dim() == 0  # Scalar
        assert result.item() > 0

    def test_multi_batch_accumulation(self, simple_2d_vectors):
        """Test that multi-batch gives same result as single large batch."""
        preds, target = simple_2d_vectors
        metric = NormalizedVectorNormError(vector_dim=-1, norm_ord=2)

        # Process as two separate batches
        metric.update(preds[:1], target[:1])
        metric.update(preds[1:], target[1:])
        multi_batch_result = metric.compute()

        # Process as single batch
        metric_single = NormalizedVectorNormError(vector_dim=-1, norm_ord=2)
        metric_single.update(preds, target)
        single_batch_result = metric_single.compute()

        assert torch.allclose(multi_batch_result, single_batch_result)

    def test_consistent_with_functional(self, simple_2d_vectors):
        """Test wrapper gives same result as functional implementation."""
        preds, target = simple_2d_vectors

        # TorchMetrics wrapper
        metric = NormalizedVectorNormError(vector_dim=-1, norm_ord=2)
        metric.update(preds, target)
        wrapper_result = metric.compute()

        # Functional implementation
        functional_result = normalized_vector_norm_error(preds, target, ord=2, dim=-1, eps=1e-8)

        assert torch.allclose(wrapper_result, functional_result, atol=1e-6)

    def test_reset(self, simple_2d_vectors):
        """Test reset clears state."""
        preds, target = simple_2d_vectors
        metric = NormalizedVectorNormError(vector_dim=-1, norm_ord=2)

        # First update
        metric.update(preds, target)
        first_result = metric.compute()

        # Reset
        metric.reset()

        # State should be zero
        assert metric.sum_errors == 0
        assert metric.total == 0

        # Second update should give same result as first
        metric.update(preds, target)
        second_result = metric.compute()

        assert torch.allclose(first_result, second_result)

    def test_different_norm_orders(self, simple_2d_vectors):
        """Test different norm orders (L1, L2, Linf)."""
        preds, target = simple_2d_vectors

        metric_l1 = NormalizedVectorNormError(norm_ord=1)
        metric_l2 = NormalizedVectorNormError(norm_ord=2)

        metric_l1.update(preds, target)
        metric_l2.update(preds, target)

        error_l1 = metric_l1.compute()
        error_l2 = metric_l2.compute()

        # Both should be positive scalars
        assert error_l1.item() > 0
        assert error_l2.item() > 0
        # Results should differ
        assert not torch.allclose(error_l1, error_l2)

    def test_batch_with_3d_vectors(self):
        """Test with 3D vectors."""
        preds = torch.randn(16, 8)  # (batch, features)
        target = torch.randn(16, 8)

        metric = NormalizedVectorNormError(vector_dim=-1, norm_ord=2)
        metric.update(preds, target)
        result = metric.compute()

        assert result.dim() == 0
        assert not torch.isnan(result)

    def test_invalid_1d_input_fails(self):
        """Test that 1D input raises error."""
        metric = NormalizedVectorNormError(vector_dim=-1, norm_ord=2)
        preds = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.1, 1.9, 3.1])

        with pytest.raises(ValueError, match="at least 2D"):
            metric.update(preds, target)


# ============================================================================
# 2. TEMPORAL DERIVATIVE ERROR TESTS
# ============================================================================


class TestTemporalDerivativeError:
    """Test TemporalDerivativeError wrapper."""

    def test_single_batch_first_derivative(self, temporal_sequence_3d):
        """Test first derivative with single batch."""
        preds, target = temporal_sequence_3d
        metric = TemporalDerivativeError(n=1, derivative_dim=1)

        metric.update(preds, target)
        result = metric.compute()

        assert result.dim() == 0  # Scalar
        assert result.item() >= 0  # Squared error is non-negative

    def test_single_batch_second_derivative(self, temporal_sequence_3d):
        """Test second derivative with single batch."""
        preds, target = temporal_sequence_3d
        metric = TemporalDerivativeError(n=2, derivative_dim=1)

        metric.update(preds, target)
        result = metric.compute()

        assert result.dim() == 0
        assert result.item() >= 0

    def test_multi_batch_accumulation(self, temporal_sequence_3d):
        """Test multi-batch gives same result as single batch."""
        preds, target = temporal_sequence_3d
        metric = TemporalDerivativeError(n=1, derivative_dim=1)

        # Process as two separate batches
        metric.update(preds[:1], target[:1])
        metric.update(preds[1:], target[1:])
        multi_batch_result = metric.compute()

        # Process as single batch
        metric_single = TemporalDerivativeError(n=1, derivative_dim=1)
        metric_single.update(preds, target)
        single_batch_result = metric_single.compute()

        assert torch.allclose(multi_batch_result, single_batch_result)

    def test_consistent_with_functional(self, temporal_sequence_3d):
        """Test wrapper gives same result as functional implementation."""
        preds, target = temporal_sequence_3d

        # TorchMetrics wrapper
        metric = TemporalDerivativeError(n=1, derivative_dim=1)
        metric.update(preds, target)
        wrapper_result = metric.compute()

        # Functional implementation
        functional_result = temporal_derivative_error(preds, target, n=1, derivative_dim=1)

        assert torch.allclose(wrapper_result, functional_result, atol=1e-6)

    def test_reset(self, temporal_sequence_3d):
        """Test reset clears state."""
        preds, target = temporal_sequence_3d
        metric = TemporalDerivativeError(n=1, derivative_dim=1)

        # First update
        metric.update(preds, target)
        first_result = metric.compute()

        # Reset
        metric.reset()

        # State should be zero
        assert metric.sum_squared_errors == 0
        assert metric.total == 0

        # Second update
        metric.update(preds, target)
        second_result = metric.compute()

        assert torch.allclose(first_result, second_result)

    def test_invalid_2d_input_fails(self):
        """Test that 2D input raises error."""
        metric = TemporalDerivativeError(n=1, derivative_dim=1)
        preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.1, 1.9], [3.1, 3.9]])

        with pytest.raises(ValueError, match="3D input"):
            metric.update(preds, target)

    def test_insufficient_time_steps_fails(self):
        """Test that T < n+1 raises error."""
        metric = TemporalDerivativeError(n=2, derivative_dim=1)
        # Only 2 time steps, need 3 for 2nd derivative
        preds = torch.tensor([[[1.0], [2.0]]])  # (1, 2, 1)
        target = torch.tensor([[[1.1], [1.9]]])

        with pytest.raises(ValueError, match="too small"):
            metric.update(preds, target)

    def test_larger_sequence(self):
        """Test with larger temporal sequence."""
        preds = torch.randn(8, 20, 5)  # (batch=8, time=20, features=5)
        target = torch.randn(8, 20, 5)

        metric = TemporalDerivativeError(n=1, derivative_dim=1)
        metric.update(preds, target)
        result = metric.compute()

        assert result.dim() == 0
        assert not torch.isnan(result)


# ============================================================================
# 3. METRIC COLLECTION TESTS (CRITICAL!)
# ============================================================================


class TestMetricCollectionIntegration:
    """Test integration with torchmetrics.MetricCollection.

    This is the CRITICAL functionality we need for MLflow logging.
    """

    def test_single_custom_metric_in_collection(self, simple_2d_vectors):
        """Test single custom metric in MetricCollection."""
        preds, target = simple_2d_vectors

        # Create collection with single custom metric
        metrics = MetricCollection([NormalizedVectorNormError(vector_dim=-1, norm_ord=2)])

        # Update and compute
        metrics.update(preds, target)
        results = metrics.compute()

        # Should return dict with metric results
        assert isinstance(results, dict)
        assert len(results) == 1
        # Get first value
        result_value = list(results.values())[0]
        assert result_value.item() > 0

    def test_multiple_custom_metrics_in_collection(self, simple_2d_vectors):
        """Test multiple custom metrics in MetricCollection."""
        preds, target = simple_2d_vectors

        # Create collection with multiple metrics
        metrics = MetricCollection({
            "norm_l1": NormalizedVectorNormError(norm_ord=1),
            "norm_l2": NormalizedVectorNormError(norm_ord=2),
        })

        metrics.update(preds, target)
        results = metrics.compute()

        # Should have both metrics
        assert "norm_l1" in results
        assert "norm_l2" in results
        assert results["norm_l1"].item() > 0
        assert results["norm_l2"].item() > 0

    def test_temporal_metric_in_collection(self, temporal_sequence_3d):
        """Test temporal metric in MetricCollection."""
        preds, target = temporal_sequence_3d

        metrics = MetricCollection({
            "velocity_error": TemporalDerivativeError(n=1, derivative_dim=1),
            "accel_error": TemporalDerivativeError(n=2, derivative_dim=1),
        })

        metrics.update(preds, target)
        results = metrics.compute()

        assert "velocity_error" in results
        assert "accel_error" in results
        assert results["velocity_error"].item() >= 0
        assert results["accel_error"].item() >= 0

    def test_mixed_metrics_in_collection(self):
        """Test mixing standard and custom metrics in collection."""
        from torchmetrics.regression import MeanSquaredError

        preds_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target_2d = torch.tensor([[1.1, 1.9], [3.1, 3.9]])

        # Mix standard torchmetrics with custom
        metrics = MetricCollection({
            "mse": MeanSquaredError(),
            "norm_error": NormalizedVectorNormError(vector_dim=-1, norm_ord=2),
        })

        metrics.update(preds_2d, target_2d)
        results = metrics.compute()

        assert "mse" in results
        assert "norm_error" in results
        assert results["mse"].item() > 0
        assert results["norm_error"].item() > 0

    def test_metric_collection_reset(self, simple_2d_vectors):
        """Test MetricCollection reset works for custom metrics."""
        preds, target = simple_2d_vectors

        metrics = MetricCollection([NormalizedVectorNormError(norm_ord=2)])

        # First batch
        metrics.update(preds, target)
        first_results = metrics.compute()

        # Reset
        metrics.reset()

        # Second batch should give same results
        metrics.update(preds, target)
        second_results = metrics.compute()

        first_value = list(first_results.values())[0]
        second_value = list(second_results.values())[0]
        assert torch.allclose(first_value, second_value)


# ============================================================================
# 4. EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_target_norms_with_epsilon(self):
        """Test epsilon handling with zero target norms."""
        preds = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        target = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

        metric = NormalizedVectorNormError(eps=1e-8)
        metric.update(preds, target)
        result = metric.compute()

        # Should not be NaN or Inf
        assert not torch.isnan(result)
        assert not torch.isinf(result)

    def test_single_sample(self):
        """Test with batch size of 1."""
        preds = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.1, 1.9, 3.1]])

        metric = NormalizedVectorNormError()
        metric.update(preds, target)
        result = metric.compute()

        assert result.dim() == 0
        assert result.item() > 0

    def test_large_batch(self):
        """Test with large batch."""
        preds = torch.randn(1000, 20)
        target = torch.randn(1000, 20)

        metric = NormalizedVectorNormError()
        metric.update(preds, target)
        result = metric.compute()

        assert not torch.isnan(result)
        assert not torch.isinf(result)
