"""Tests for shared functional loss/metric implementations.

Tests differentiability and correctness of functions in dlkit.core.training.functional.
"""

import pytest
import torch
from torch import Tensor

from dlkit.core.training.functional import (
    huber_loss,
    log_cosh_loss,
    mae,
    mse,
    normalized_mse,
    normalized_vector_norm_loss,
    quantile_loss,
    relative_energy_norm_loss,
    smooth_l1_loss,
    vector_norm_loss,
)
from dlkit.core.training.metrics.functional import (
    _relative_energy_norm_compute,
    _relative_energy_norm_update,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def simple_tensors():
    """Simple pred/target tensors."""
    preds = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = torch.tensor([1.1, 1.9, 3.1])
    return preds, target


@pytest.fixture
def vector_tensors():
    """2D vector tensors for normalized vector norm loss."""
    preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target = torch.tensor([[1.1, 1.9], [3.1, 3.9]])
    return preds, target


# ============================================================================
# DIFFERENTIABILITY TESTS
# ============================================================================


class TestDifferentiability:
    """Test that all loss functions are differentiable."""

    def test_mse_differentiable(self, simple_tensors):
        """Test MSE is differentiable."""
        preds, target = simple_tensors
        loss = mse(preds, target)
        loss.backward()

        assert preds.grad is not None
        assert not torch.isnan(preds.grad).any()

    def test_mae_differentiable(self, simple_tensors):
        """Test MAE is differentiable."""
        preds, target = simple_tensors
        loss = mae(preds, target)
        loss.backward()

        assert preds.grad is not None
        assert not torch.isnan(preds.grad).any()

    def test_huber_loss_differentiable(self, simple_tensors):
        """Test Huber loss is differentiable."""
        preds, target = simple_tensors
        loss = huber_loss(preds, target, delta=1.0)
        loss.backward()

        assert preds.grad is not None
        assert not torch.isnan(preds.grad).any()

    def test_smooth_l1_differentiable(self, simple_tensors):
        """Test Smooth L1 is differentiable."""
        preds, target = simple_tensors
        loss = smooth_l1_loss(preds, target, beta=1.0)
        loss.backward()

        assert preds.grad is not None
        assert not torch.isnan(preds.grad).any()

    def test_log_cosh_differentiable(self, simple_tensors):
        """Test log-cosh is differentiable."""
        preds, target = simple_tensors
        loss = log_cosh_loss(preds, target)
        loss.backward()

        assert preds.grad is not None
        assert not torch.isnan(preds.grad).any()

    def test_quantile_loss_differentiable(self, simple_tensors):
        """Test quantile loss is differentiable."""
        preds, target = simple_tensors
        loss = quantile_loss(preds, target, quantile=0.5)
        loss.backward()

        assert preds.grad is not None
        assert not torch.isnan(preds.grad).any()

    def test_normalized_mse_differentiable(self, simple_tensors):
        """Test normalized MSE is differentiable."""
        preds, target = simple_tensors
        loss = normalized_mse(preds, target, normalization="variance")
        loss.backward()

        assert preds.grad is not None
        assert not torch.isnan(preds.grad).any()

    def test_normalized_vector_norm_loss_differentiable(self, vector_tensors):
        """Test normalized vector norm loss is differentiable."""
        preds, target = vector_tensors
        loss = normalized_vector_norm_loss(preds, target, ord=2, dim=-1)
        loss.backward()

        assert preds.grad is not None
        assert not torch.isnan(preds.grad).any()


# ============================================================================
# CORRECTNESS TESTS
# ============================================================================


class TestCorrectness:
    """Test correct values for loss functions."""

    def test_mse_value(self):
        """Test MSE computes correct value."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = mse(preds, target)

        assert torch.allclose(loss, torch.tensor(0.0))

    def test_mae_value(self):
        """Test MAE computes correct value."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        loss = mae(preds, target)

        assert torch.allclose(loss, torch.tensor(1.0))

    def test_huber_loss_quadratic_region(self):
        """Test Huber loss in quadratic region (error < delta)."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.1, 2.1, 3.1])
        loss = huber_loss(preds, target, delta=1.0)

        # Small errors, should be close to MSE
        mse_loss = mse(preds, target)
        assert loss < mse_loss * 1.1  # Within 10%

    def test_huber_loss_linear_region(self):
        """Test Huber loss in linear region (error > delta)."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([5.0, 6.0, 7.0])
        loss = huber_loss(preds, target, delta=1.0)

        # Large errors, should be less sensitive than MSE
        mse_loss = mse(preds, target)
        assert loss < mse_loss  # Huber is more robust

    def test_quantile_loss_median(self):
        """Test quantile loss at 0.5 equals MAE/2."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 3.0, 4.0])

        q_loss = quantile_loss(preds, target, quantile=0.5)
        mae_loss = mae(preds, target)

        # Quantile loss at 0.5 is MAE/2 due to symmetric weighting
        assert torch.allclose(q_loss, mae_loss / 2)

    def test_normalized_mse_variance(self):
        """Test normalized MSE by variance."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.1, 2.1, 3.1])

        loss = normalized_mse(preds, target, normalization="variance")
        expected = mse(preds, target) / (torch.var(target) + 1e-8)

        assert torch.allclose(loss, expected)

    def test_vector_norm_loss_defaults_to_mean_aggregation(self):
        """Default aggregator should match explicit torch.mean."""
        preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 2.5], [2.5, 3.5]])

        implicit = vector_norm_loss(preds, target, ord=2, dim=-1)
        explicit = vector_norm_loss(preds, target, ord=2, dim=-1, aggregator=torch.mean)

        assert torch.allclose(implicit, explicit)

    def test_relative_energy_norm_loss_parity_with_update_compute_split(self):
        """Direct relative energy loss should match update/compute split behavior."""
        preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 2.5], [2.5, 3.5]])
        matrix = torch.eye(2).unsqueeze(0)

        direct = relative_energy_norm_loss(preds, target, matrix, eps=1e-8)
        per_sample = _relative_energy_norm_update(preds, target, matrix, eps=1e-8)
        split = _relative_energy_norm_compute(per_sample.sum(), per_sample.numel())

        assert torch.allclose(direct, split)

    def test_relative_energy_norm_loss_sparse_matches_dense(self):
        """Sparse and dense matrices should produce the same relative energy loss."""
        preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 2.5], [2.5, 3.5]])
        dense_matrix = torch.eye(2).unsqueeze(0)
        sparse_matrix = dense_matrix.to_sparse_coo()

        dense = relative_energy_norm_loss(preds, target, dense_matrix, eps=1e-8)
        sparse = relative_energy_norm_loss(preds, target, sparse_matrix, eps=1e-8)

        assert torch.allclose(dense, sparse)

    def test_relative_energy_norm_loss_sparse_per_sample_matches_dense(self):
        """Sparse per-sample (B, D, D) relative energy loss should match dense."""
        preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 2.5], [2.5, 3.5]])
        dense_batch = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 1.0]],
                [[3.0, 0.2], [0.2, 2.0]],
            ]
        )
        sparse_batch = dense_batch.to_sparse_coo()

        dense = relative_energy_norm_loss(preds, target, dense_batch, eps=1e-8)
        sparse = relative_energy_norm_loss(preds, target, sparse_batch, eps=1e-8)

        assert torch.allclose(dense, sparse)


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_target_mse(self):
        """Test MSE with zero target."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([0.0, 0.0, 0.0])
        loss = mse(preds, target)

        # Should work without NaN
        assert not torch.isnan(loss)

    def test_quantile_invalid_value(self):
        """Test quantile loss raises on invalid quantile."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.1, 2.1, 3.1])

        with pytest.raises(ValueError, match="Quantile must be in"):
            quantile_loss(preds, target, quantile=1.5)

    def test_normalized_mse_invalid_method(self):
        """Test normalized MSE raises on invalid normalization."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.1, 2.1, 3.1])

        with pytest.raises(ValueError, match="Invalid normalization"):
            normalized_mse(preds, target, normalization="invalid")

    def test_normalized_vector_norm_with_zero_target(self, vector_tensors):
        """Test normalized vector norm handles zero targets with eps."""
        preds = torch.tensor([[1.0, 2.0]], requires_grad=True)
        target = torch.tensor([[0.0, 0.0]])

        loss = normalized_vector_norm_loss(preds, target, eps=1e-8)

        # Should not be NaN or Inf due to eps
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Test integration with training workflows."""

    def test_use_as_loss_in_backprop(self, simple_tensors):
        """Test function can be used as loss in training loop."""
        preds, target = simple_tensors

        # Simulate training step
        optimizer = torch.optim.SGD([preds], lr=0.01)
        optimizer.zero_grad()

        loss = mse(preds, target)
        loss.backward()
        optimizer.step()

        # Params should have changed
        assert preds.grad is not None

    def test_multiple_losses_composition(self, simple_tensors):
        """Test combining multiple losses."""
        preds, target = simple_tensors

        loss = 0.7 * mse(preds, target) + 0.3 * mae(preds, target)
        loss.backward()

        assert preds.grad is not None
        assert not torch.isnan(preds.grad).any()

    def test_batch_processing(self):
        """Test with batch of samples."""
        batch_size = 16
        preds = torch.randn(batch_size, 10, requires_grad=True)
        target = torch.randn(batch_size, 10)

        loss = huber_loss(preds, target)
        loss.backward()

        assert preds.grad.shape == preds.shape
