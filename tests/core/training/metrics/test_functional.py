"""Comprehensive tests for functional metrics module.

Tests all pure functional metric implementations with focus on:
    - Shape transformations
    - Edge cases (zero division, dimension bounds, temporal constraints)
    - Composition patterns
    - Different aggregators and norm orders
"""

from functools import partial

import pytest
import torch
from torch import Tensor

from dlkit.core.training.metrics.functional import (
    # Primitives
    compute_error_vectors,
    compute_vector_norm,
    safe_divide,
    apply_aggregation,
    # Vector metrics
    normalized_vector_norm_error,
    normalized_l1_error,
    normalized_l2_error,
    normalized_linf_error,
    # Temporal metrics
    compute_temporal_derivative,
    temporal_derivative_error,
    first_derivative_error,
    second_derivative_error,
    # Update/compute split
    _normalized_vector_norm_update,
    _normalized_vector_norm_compute,
    _temporal_derivative_update,
    _temporal_derivative_compute,
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
def batch_3d_vectors():
    """Batch of 3D vectors (B=2, D=3)."""
    preds = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    target = torch.tensor([[1.1, 1.9, 3.1], [3.9, 5.1, 5.9]])
    return preds, target


@pytest.fixture
def temporal_sequence_3d():
    """Temporal sequence with shape (B=2, T=4, D=2)."""
    preds = torch.tensor([
        [[0.0, 0.0], [1.0, 0.5], [2.5, 1.5], [4.0, 2.0]],
        [[0.0, 1.0], [1.5, 2.0], [3.0, 3.5], [4.5, 5.0]]
    ])
    target = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    ])
    return preds, target


@pytest.fixture
def zero_target_vectors():
    """Vectors with zero target norms for testing epsilon handling."""
    preds = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    target = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    return preds, target


# ============================================================================
# 1. PRIMITIVES TESTS
# ============================================================================


class TestPrimitives:
    """Test composable building blocks."""

    def test_compute_error_vectors(self, simple_2d_vectors):
        """Test basic error computation."""
        preds, target = simple_2d_vectors
        error = compute_error_vectors(preds, target)

        expected = torch.tensor([[0.0, -1.0], [-2.0, 2.0]])
        assert torch.allclose(error, expected)

    def test_compute_error_vectors_shape_mismatch(self):
        """Test error raised on shape mismatch."""
        preds = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_error_vectors(preds, target)

    def test_compute_vector_norm_l2(self, simple_2d_vectors):
        """Test L2 norm computation."""
        preds, _ = simple_2d_vectors
        norms = compute_vector_norm(preds, ord=2, dim=-1)

        expected = torch.tensor([1.0, 2.0])
        assert torch.allclose(norms, expected)

    def test_compute_vector_norm_l1(self, simple_2d_vectors):
        """Test L1 norm computation."""
        preds, _ = simple_2d_vectors
        norms = compute_vector_norm(preds, ord=1, dim=-1)

        expected = torch.tensor([1.0, 2.0])
        assert torch.allclose(norms, expected)

    def test_compute_vector_norm_invalid_dim(self, simple_2d_vectors):
        """Test error raised on invalid dimension."""
        preds, _ = simple_2d_vectors

        with pytest.raises(ValueError, match="out of bounds"):
            compute_vector_norm(preds, ord=2, dim=5)

    def test_safe_divide_normal(self):
        """Test safe division without zeros."""
        num = torch.tensor([1.0, 2.0, 3.0])
        denom = torch.tensor([2.0, 4.0, 6.0])
        result = safe_divide(num, denom)

        expected = torch.tensor([0.5, 0.5, 0.5])
        assert torch.allclose(result, expected)

    def test_safe_divide_with_zero(self):
        """Test safe division handles zero denominator."""
        num = torch.tensor([1.0, 2.0])
        denom = torch.tensor([0.0, 2.0])
        result = safe_divide(num, denom, eps=1e-8)

        # First element should be num / eps, second is 1.0
        assert result[1] == 1.0
        assert result[0] > 1e7  # Very large due to division by small eps

    def test_apply_aggregation_mean(self):
        """Test aggregation with mean."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = apply_aggregation(tensor, torch.mean)

        assert torch.allclose(result, torch.tensor(2.5))

    def test_apply_aggregation_sum(self):
        """Test aggregation with sum."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = apply_aggregation(tensor, torch.sum)

        assert torch.allclose(result, torch.tensor(10.0))

    def test_apply_aggregation_with_partial(self):
        """Test aggregation with functools.partial for kwargs."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = apply_aggregation(tensor, partial(torch.mean, dim=0))

        expected = torch.tensor([2.0, 3.0])
        assert torch.allclose(result, expected)


# ============================================================================
# 2. VECTOR METRICS TESTS
# ============================================================================


class TestVectorMetrics:
    """Test vector norm metrics."""

    def test_normalized_vector_norm_error_l2(self, simple_2d_vectors):
        """Test L2 normalized vector norm error."""
        preds, target = simple_2d_vectors
        error = normalized_vector_norm_error(preds, target, ord=2, dim=-1)

        # Vector 1: ||[0, -1]|| / ||[1, 1]|| = 1.0 / sqrt(2) ≈ 0.7071
        # Vector 2: ||[-2, 2]|| / ||[2, 0]|| = sqrt(8) / 2 ≈ 1.4142
        # Mean ≈ 1.0607
        assert error.item() > 1.0 and error.item() < 1.1

    def test_normalized_vector_norm_error_l1(self, simple_2d_vectors):
        """Test L1 normalized vector norm error."""
        preds, target = simple_2d_vectors
        error = normalized_l1_error(preds, target, dim=-1)

        # Using convenience partial
        assert error.dim() == 0  # Scalar output

    def test_normalized_vector_norm_error_3d(self, batch_3d_vectors):
        """Test with 3D vectors."""
        preds, target = batch_3d_vectors
        error = normalized_vector_norm_error(preds, target, ord=2, dim=-1)

        assert error.dim() == 0  # Scalar after aggregation
        assert error.item() > 0

    def test_normalized_vector_norm_error_1d_fails(self):
        """Test that 1D input fails validation."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.1, 1.9, 3.1])

        with pytest.raises(ValueError, match="at least 2D"):
            normalized_vector_norm_error(preds, target)

    def test_normalized_vector_norm_error_zero_target(self, zero_target_vectors):
        """Test epsilon handling with zero target norms."""
        preds, target = zero_target_vectors
        error = normalized_vector_norm_error(preds, target, ord=2, dim=-1, eps=1e-8)

        # Should not raise, epsilon prevents division by zero
        assert not torch.isnan(error)
        assert not torch.isinf(error)

    def test_normalized_vector_norm_error_custom_aggregator(self, simple_2d_vectors):
        """Test with custom aggregator (sum instead of mean)."""
        preds, target = simple_2d_vectors
        error = normalized_vector_norm_error(
            preds, target, ord=2, dim=-1, aggregator=torch.sum
        )

        # Should be sum of per-sample errors, not mean
        error_mean = normalized_vector_norm_error(preds, target, ord=2, dim=-1)
        assert error > error_mean

    def test_normalized_linf_error(self, simple_2d_vectors):
        """Test Linf norm variant."""
        preds, target = simple_2d_vectors
        error = normalized_linf_error(preds, target, dim=-1)

        assert error.dim() == 0  # Scalar
        assert error.item() > 0


# ============================================================================
# 3. TEMPORAL METRICS TESTS
# ============================================================================


class TestTemporalMetrics:
    """Test temporal derivative metrics with 3D inputs."""

    def test_compute_temporal_derivative_first_order(self, temporal_sequence_3d):
        """Test first derivative computation."""
        preds, _ = temporal_sequence_3d
        derivative = compute_temporal_derivative(preds, n=1, derivative_dim=1)

        # Shape should be (B, T-1, D) = (2, 3, 2)
        assert derivative.shape == (2, 3, 2)

    def test_compute_temporal_derivative_second_order(self, temporal_sequence_3d):
        """Test second derivative computation."""
        preds, _ = temporal_sequence_3d
        derivative = compute_temporal_derivative(preds, n=2, derivative_dim=1)

        # Shape should be (B, T-2, D) = (2, 2, 2)
        assert derivative.shape == (2, 2, 2)

    def test_compute_temporal_derivative_not_3d_fails(self):
        """Test that non-3D input fails validation."""
        tensor_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="3D input"):
            compute_temporal_derivative(tensor_2d, n=1, derivative_dim=1)

    def test_compute_temporal_derivative_insufficient_time_steps(self):
        """Test that T < n+1 fails validation."""
        # Only 2 time steps, need 3 for 2nd derivative
        tensor = torch.tensor([[[1.0], [2.0]]])  # (1, 2, 1)

        with pytest.raises(ValueError, match="too small"):
            compute_temporal_derivative(tensor, n=2, derivative_dim=1)

    def test_compute_temporal_derivative_boundary_case(self):
        """Test boundary case: T = n+1 (minimum valid)."""
        # Exactly 2 time steps for 1st derivative
        tensor = torch.tensor([[[1.0], [2.0]]])  # (1, 2, 1)
        derivative = compute_temporal_derivative(tensor, n=1, derivative_dim=1)

        # Should work and produce shape (1, 1, 1)
        assert derivative.shape == (1, 1, 1)
        assert torch.allclose(derivative, torch.tensor([[[1.0]]]))

    def test_temporal_derivative_error_first_order(self, temporal_sequence_3d):
        """Test first derivative error (velocity error)."""
        preds, target = temporal_sequence_3d
        error = temporal_derivative_error(preds, target, n=1, derivative_dim=1)

        assert error.dim() == 0  # Scalar after aggregation
        assert error.item() >= 0  # Squared error is non-negative

    def test_temporal_derivative_error_second_order(self, temporal_sequence_3d):
        """Test second derivative error (acceleration error)."""
        preds, target = temporal_sequence_3d
        error = temporal_derivative_error(preds, target, n=2, derivative_dim=1)

        assert error.dim() == 0  # Scalar
        assert error.item() >= 0

    def test_first_derivative_error_convenience(self, temporal_sequence_3d):
        """Test convenience partial for first derivative."""
        preds, target = temporal_sequence_3d
        error = first_derivative_error(preds, target, derivative_dim=1)

        # Should be same as calling with n=1
        error_explicit = temporal_derivative_error(preds, target, n=1, derivative_dim=1)
        assert torch.allclose(error, error_explicit)

    def test_second_derivative_error_convenience(self, temporal_sequence_3d):
        """Test convenience partial for second derivative."""
        preds, target = temporal_sequence_3d
        error = second_derivative_error(preds, target, derivative_dim=1)

        # Should be same as calling with n=2
        error_explicit = temporal_derivative_error(preds, target, n=2, derivative_dim=1)
        assert torch.allclose(error, error_explicit)

    def test_temporal_derivative_error_custom_aggregator(self, temporal_sequence_3d):
        """Test with custom aggregator."""
        preds, target = temporal_sequence_3d
        error_sum = temporal_derivative_error(
            preds, target, n=1, derivative_dim=1, aggregator=torch.sum
        )
        error_mean = temporal_derivative_error(
            preds, target, n=1, derivative_dim=1, aggregator=torch.mean
        )

        # Sum should be larger than mean
        assert error_sum > error_mean


# ============================================================================
# 4. UPDATE/COMPUTE SPLIT TESTS
# ============================================================================


class TestUpdateComputeSplit:
    """Test update/compute functions for torchmetrics integration."""

    def test_normalized_vector_norm_update(self, simple_2d_vectors):
        """Test update function returns per-sample errors."""
        preds, target = simple_2d_vectors
        errors = _normalized_vector_norm_update(preds, target, ord=2, dim=-1, eps=1e-8)

        # Should have one error per sample
        assert errors.shape == (2,)
        assert torch.all(errors >= 0)

    def test_normalized_vector_norm_compute(self):
        """Test compute function calculates mean."""
        sum_errors = torch.tensor(10.0)
        total = 5
        mean_error = _normalized_vector_norm_compute(sum_errors, total)

        assert torch.allclose(mean_error, torch.tensor(2.0))

    def test_normalized_vector_norm_split_equals_direct(self, simple_2d_vectors):
        """Test update/compute split gives same result as direct function."""
        preds, target = simple_2d_vectors

        # Direct functional call
        direct_result = normalized_vector_norm_error(preds, target, ord=2, dim=-1)

        # Update/compute split
        errors = _normalized_vector_norm_update(preds, target, ord=2, dim=-1, eps=1e-8)
        sum_errors = errors.sum()
        total = errors.numel()
        split_result = _normalized_vector_norm_compute(sum_errors, total)

        assert torch.allclose(direct_result, split_result)

    def test_temporal_derivative_update(self, temporal_sequence_3d):
        """Test temporal update returns per-element squared errors."""
        preds, target = temporal_sequence_3d
        squared_errors = _temporal_derivative_update(preds, target, n=1, derivative_dim=1)

        # Shape should be (B, T-1, D) = (2, 3, 2)
        assert squared_errors.shape == (2, 3, 2)
        assert torch.all(squared_errors >= 0)  # Squared, so non-negative

    def test_temporal_derivative_compute(self):
        """Test temporal compute calculates mean."""
        sum_squared = torch.tensor(20.0)
        total = 10
        mean_error = _temporal_derivative_compute(sum_squared, total)

        assert torch.allclose(mean_error, torch.tensor(2.0))

    def test_temporal_derivative_split_equals_direct(self, temporal_sequence_3d):
        """Test update/compute split gives same result as direct function."""
        preds, target = temporal_sequence_3d

        # Direct functional call
        direct_result = temporal_derivative_error(preds, target, n=1, derivative_dim=1)

        # Update/compute split
        squared_errors = _temporal_derivative_update(preds, target, n=1, derivative_dim=1)
        sum_squared = squared_errors.sum()
        total = squared_errors.numel()
        split_result = _temporal_derivative_compute(sum_squared, total)

        assert torch.allclose(direct_result, split_result)


# ============================================================================
# 5. EDGE CASES AND INTEGRATION TESTS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sample_batch(self):
        """Test with batch size of 1."""
        preds = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.1, 1.9, 3.1]])
        error = normalized_vector_norm_error(preds, target, ord=2, dim=-1)

        assert error.dim() == 0  # Should still be scalar
        assert error.item() > 0

    def test_large_batch(self):
        """Test with larger batch."""
        preds = torch.randn(100, 10)
        target = torch.randn(100, 10)
        error = normalized_vector_norm_error(preds, target, ord=2, dim=-1)

        assert error.dim() == 0
        assert not torch.isnan(error)

    def test_temporal_with_many_features(self):
        """Test temporal metric with more feature dimensions."""
        preds = torch.randn(8, 20, 16)  # (B=8, T=20, D=16)
        target = torch.randn(8, 20, 16)
        error = temporal_derivative_error(preds, target, n=1, derivative_dim=1)

        assert error.dim() == 0
        assert not torch.isnan(error)

    def test_composition_via_partial(self, simple_2d_vectors):
        """Test function composition with functools.partial."""
        preds, target = simple_2d_vectors

        # Create custom metric via partial composition
        my_l1_sum_metric = partial(
            normalized_vector_norm_error,
            ord=1,
            dim=-1,
            aggregator=torch.sum
        )

        result = my_l1_sum_metric(preds, target)
        assert result.dim() == 0
