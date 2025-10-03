"""Tests for metric implementations."""

import pytest
import torch
from torch import Tensor

from dlkit.core.training.metrics import (
    MeanSquaredErrorMetric,
    MeanAbsoluteErrorMetric,
    RootMeanSquaredErrorMetric,
    NormalizedVectorNormErrorMetric,
    MSEOverVarianceMetric,
    TemporalDerivativeMetric,
)


class TestMeanSquaredErrorMetric:
    """Tests for MSE metric."""

    def test_mse_basic_computation(self, sample_predictions, sample_targets):
        """Test basic MSE computation."""
        metric = MeanSquaredErrorMetric()
        result = metric.compute(sample_predictions, sample_targets)

        # Manual calculation
        diff = sample_predictions - sample_targets
        expected = torch.mean(diff**2)

        assert torch.allclose(result, expected, atol=1e-6)

    def test_mse_with_custom_aggregator(self, sample_predictions, sample_targets):
        """Test MSE with custom aggregator."""
        from dlkit.core.training.metrics import SumAggregator

        metric = MeanSquaredErrorMetric(aggregator=SumAggregator())
        result = metric.compute(sample_predictions, sample_targets)

        # Manual calculation with sum instead of mean
        diff = sample_predictions - sample_targets
        expected = torch.sum(diff**2)

        assert torch.allclose(result, expected, atol=1e-6)

    def test_mse_shape_mismatch_error(self):
        """Test that shape mismatch raises ValueError."""
        metric = MeanSquaredErrorMetric()
        predictions = torch.tensor([[1.0, 2.0]])
        targets = torch.tensor([[1.0], [2.0]])

        with pytest.raises(ValueError, match="Shape mismatch"):
            metric.compute(predictions, targets)

    def test_mse_metadata(self):
        """Test metric metadata."""
        metric = MeanSquaredErrorMetric()
        metadata = metric.metadata

        assert metadata["name"] == "mse"
        assert metadata["aggregator"] == "mean"
        assert "parameters" in metadata


class TestNormalizedVectorNormErrorMetric:
    """Tests for normalized vector norm error metric."""

    def test_normalized_vector_norm_basic(self, sample_2d_vectors):
        """Test basic normalized vector norm error computation."""
        predictions, targets = sample_2d_vectors
        metric = NormalizedVectorNormErrorMetric(vector_dim=-1, norm_ord=2)

        result = metric.compute(predictions, targets)

        # Manual calculation
        error_vectors = predictions - targets
        error_norms = torch.linalg.vector_norm(error_vectors, ord=2, dim=-1)
        target_norms = torch.linalg.vector_norm(targets, ord=2, dim=-1)
        normalized_errors = error_norms / (target_norms + 1e-8)
        expected = torch.mean(normalized_errors)

        assert torch.allclose(result, expected, atol=1e-6)

    def test_normalized_vector_norm_l1(self, sample_2d_vectors):
        """Test normalized vector norm error with L1 norm."""
        predictions, targets = sample_2d_vectors
        metric = NormalizedVectorNormErrorMetric(vector_dim=-1, norm_ord=1)

        result = metric.compute(predictions, targets)

        # Manual calculation with L1 norm
        error_vectors = predictions - targets
        error_norms = torch.linalg.vector_norm(error_vectors, ord=1, dim=-1)
        target_norms = torch.linalg.vector_norm(targets, ord=1, dim=-1)
        normalized_errors = error_norms / (target_norms + 1e-8)
        expected = torch.mean(normalized_errors)

        assert torch.allclose(result, expected, atol=1e-6)

    def test_normalized_vector_norm_zero_targets(self):
        """Test numerical stability with zero targets."""
        predictions = torch.tensor([[1.0, 1.0], [0.5, 0.5]])
        targets = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        metric = NormalizedVectorNormErrorMetric(eps=1e-8)

        result = metric.compute(predictions, targets)

        # Should not raise error and return finite values
        assert torch.isfinite(result).all()

    def test_normalized_vector_norm_perfect_prediction(self):
        """Test with perfect predictions (zero error)."""
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        metric = NormalizedVectorNormErrorMetric()

        result = metric.compute(data, data)

        # Perfect prediction should give zero error
        assert torch.allclose(result, torch.tensor(0.0), atol=1e-6)

    def test_normalized_vector_norm_1d_error(self):
        """Test that 1D tensors raise ValueError."""
        metric = NormalizedVectorNormErrorMetric()
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.1, 1.9, 3.1])

        with pytest.raises(ValueError, match="Expected at least 2D tensors"):
            metric.compute(predictions, targets)

    def test_normalized_vector_norm_custom_aggregator(self, sample_2d_vectors):
        """Test with custom aggregator."""
        from dlkit.core.training.metrics import SumAggregator

        predictions, targets = sample_2d_vectors
        metric = NormalizedVectorNormErrorMetric(aggregator=SumAggregator())

        result = metric.compute(predictions, targets)

        # Should sum instead of average the normalized errors
        error_vectors = predictions - targets
        error_norms = torch.linalg.vector_norm(error_vectors, ord=2, dim=-1)
        target_norms = torch.linalg.vector_norm(targets, ord=2, dim=-1)
        normalized_errors = error_norms / (target_norms + 1e-8)
        expected = torch.sum(normalized_errors)

        assert torch.allclose(result, expected, atol=1e-6)

    def test_normalized_vector_norm_metadata(self):
        """Test metric metadata."""
        metric = NormalizedVectorNormErrorMetric(norm_ord=1, vector_dim=0)
        metadata = metric.metadata

        assert "normalized_vector_norm_error" in metadata["name"]
        assert metadata["parameters"]["norm_ord"] == 1
        assert metadata["parameters"]["vector_dim"] == 0


class TestMeanAbsoluteErrorMetric:
    """Tests for MAE metric."""

    def test_mae_basic_computation(self, sample_predictions, sample_targets):
        """Test basic MAE computation."""
        metric = MeanAbsoluteErrorMetric()
        result = metric.compute(sample_predictions, sample_targets)

        # Manual calculation
        diff = sample_predictions - sample_targets
        expected = torch.mean(torch.abs(diff))

        assert torch.allclose(result, expected, atol=1e-6)


class TestRootMeanSquaredErrorMetric:
    """Tests for RMSE metric."""

    def test_rmse_basic_computation(self, sample_predictions, sample_targets):
        """Test basic RMSE computation."""
        metric = RootMeanSquaredErrorMetric()
        result = metric.compute(sample_predictions, sample_targets)

        # Manual calculation
        diff = sample_predictions - sample_targets
        expected = torch.linalg.vector_norm(diff.flatten())

        assert torch.allclose(result, expected, atol=1e-6)


class TestMSEOverVarianceMetric:
    """Tests for MSE over variance metric."""

    def test_mse_over_var_basic(self, sample_predictions, sample_targets):
        """Test basic MSE over variance computation."""
        metric = MSEOverVarianceMetric(eps=1e-8)
        result = metric.compute(sample_predictions, sample_targets)

        # Manual calculation
        mse = torch.mean((sample_predictions - sample_targets) ** 2)
        variance = torch.var(sample_targets)
        expected = mse / (variance + 1e-8)

        assert torch.allclose(result, expected, atol=1e-6)


class TestTemporalDerivativeMetric:
    """Tests for temporal derivative metric."""

    def test_temporal_derivative_basic(self, temporal_data):
        """Test basic temporal derivative computation."""
        predictions, targets = temporal_data
        metric = TemporalDerivativeMetric(n=1, derivative_dim=-1)

        result = metric.compute(predictions, targets)

        # Manual calculation
        error = predictions - targets
        derivative_error = torch.diff(error, n=1, dim=-1)
        expected = torch.mean(derivative_error**2)

        assert torch.allclose(result, expected, atol=1e-6)

    def test_temporal_derivative_second_order(self, temporal_data):
        """Test second-order temporal derivative."""
        predictions, targets = temporal_data
        metric = TemporalDerivativeMetric(n=2, derivative_dim=-1)

        result = metric.compute(predictions, targets)

        # Manual calculation
        error = predictions - targets
        derivative_error = torch.diff(error, n=2, dim=-1)
        expected = torch.mean(derivative_error**2)

        assert torch.allclose(result, expected, atol=1e-6)


class TestMetricValidation:
    """Tests for input validation across metrics."""

    def test_non_tensor_inputs(self):
        """Test that non-tensor inputs raise TypeError."""
        metric = MeanSquaredErrorMetric()

        with pytest.raises(TypeError, match="must be torch.Tensor"):
            metric.compute([1, 2, 3], torch.tensor([1, 2, 3]))

        with pytest.raises(TypeError, match="must be torch.Tensor"):
            metric.compute(torch.tensor([1, 2, 3]), [1, 2, 3])

    def test_empty_tensors(self):
        """Test behavior with empty tensors."""
        metric = MeanSquaredErrorMetric()
        empty_predictions = torch.empty(0, 3)
        empty_targets = torch.empty(0, 3)

        # Should not raise error and return a tensor
        result = metric.compute(empty_predictions, empty_targets)
        assert result is not None
        assert isinstance(result, Tensor)
        # Empty tensor mean produces NaN, which is expected PyTorch behavior
