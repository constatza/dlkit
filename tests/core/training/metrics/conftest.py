"""Fixtures for metrics tests."""

import pytest
import torch
from torch import Tensor


@pytest.fixture
def sample_predictions() -> Tensor:
    """Sample prediction tensor for testing."""
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


@pytest.fixture
def sample_targets() -> Tensor:
    """Sample target tensor for testing."""
    return torch.tensor([[1.1, 1.9, 3.1], [3.9, 5.1, 5.9], [7.1, 7.9, 9.1]])


@pytest.fixture
def sample_2d_vectors() -> tuple[Tensor, Tensor]:
    """Sample 2D vector dataflow for normalized vector norm error testing."""
    predictions = torch.tensor([
        [1.0, 0.0],  # Predicted vector 1
        [0.0, 2.0],  # Predicted vector 2
        [1.0, 1.0],  # Predicted vector 3
    ])

    targets = torch.tensor([
        [1.0, 1.0],  # Target vector 1 (norm = sqrt(2))
        [2.0, 0.0],  # Target vector 2 (norm = 2.0)
        [0.0, 1.0],  # Target vector 3 (norm = 1.0)
    ])

    return predictions, targets


@pytest.fixture
def temporal_data() -> tuple[Tensor, Tensor]:
    """Sample temporal dataflow for time series testing."""
    # Shape: (batch, features, time)
    predictions = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0]],  # Batch 1
        [[2.0, 3.0, 4.0, 5.0]],  # Batch 2
    ])

    targets = torch.tensor([
        [[1.1, 1.9, 3.1, 3.9]],  # Batch 1
        [[2.1, 2.9, 4.1, 4.9]],  # Batch 2
    ])

    return predictions, targets


@pytest.fixture
def zero_targets() -> Tensor:
    """Zero target tensor for numerical stability testing."""
    return torch.zeros(3, 3)


@pytest.fixture
def unit_vectors() -> tuple[Tensor, Tensor]:
    """Unit vector dataflow for normalized testing."""
    predictions = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.707, 0.707],  # Approximate unit vector
    ])

    targets = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    return predictions, targets


# ============================================================================
# ENERGY NORM FIXTURES
# ============================================================================


@pytest.fixture
def batch_vectors_2d() -> tuple[Tensor, Tensor]:
    """Batch of 2D vectors: preds (B=4, D=2), targets (B=4, D=2).

    Returns:
        Tuple of (preds, targets) tensors with shape (4, 2).
    """
    preds = torch.tensor([
        [3.0, 4.0],   # L2 norm = 5.0
        [1.0, 0.0],   # L2 norm = 1.0
        [0.0, 2.0],   # L2 norm = 2.0
        [1.0, 1.0],   # L2 norm = sqrt(2)
    ])
    targets = torch.tensor([
        [3.0, 4.0],   # identical to pred 0 → zero error
        [2.0, 0.0],
        [0.0, 1.0],
        [2.0, 2.0],
    ])
    return preds, targets


@pytest.fixture
def identity_matrix_2d() -> Tensor:
    """Shared 2x2 identity matrix for energy norm tests.

    Returns:
        Identity matrix with shape (2, 2).
    """
    return torch.eye(2)


@pytest.fixture
def identity_matrix_2d_batched() -> Tensor:
    """Batched identity matrices (B=4, D=2, D=2).

    Returns:
        Batched identity matrices with shape (4, 2, 2).
    """
    return torch.eye(2).unsqueeze(0).expand(4, -1, -1).contiguous()


@pytest.fixture
def diagonal_matrix_2d() -> Tensor:
    """Diagonal 2x2 matrix with entries [2.0, 3.0] for testing quadratic form.

    When A = diag(2, 3), v^T A v = 2*v0^2 + 3*v1^2.

    Returns:
        Diagonal matrix with shape (2, 2).
    """
    return torch.diag(torch.tensor([2.0, 3.0]))


@pytest.fixture
def batch_vectors_3d() -> tuple[Tensor, Tensor]:
    """Batch of 3D vectors: preds (B=3, D=3), targets (B=3, D=3).

    Returns:
        Tuple of (preds, targets) tensors with shape (3, 3).
    """
    preds = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    targets = torch.tensor([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
    ])
    return preds, targets


@pytest.fixture
def identity_matrix_3d() -> Tensor:
    """Shared 3x3 identity matrix for energy norm tests.

    Returns:
        Identity matrix with shape (3, 3).
    """
    return torch.eye(3)


@pytest.fixture
def half_batch_split_2d(
    batch_vectors_2d: tuple[Tensor, Tensor],
    identity_matrix_2d: Tensor,
) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
    """Split the 2D batch into two half-batches for accumulation tests.

    Args:
        batch_vectors_2d: Full 2D vector batch fixture.
        identity_matrix_2d: Shared identity matrix fixture.

    Returns:
        Pair of (preds, targets, matrix) tuples for each half.
    """
    preds, targets = batch_vectors_2d
    return (
        (preds[:2], targets[:2], identity_matrix_2d),
        (preds[2:], targets[2:], identity_matrix_2d),
    )
