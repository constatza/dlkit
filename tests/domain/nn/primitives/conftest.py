import pytest
import torch
from torch import nn

from dlkit.domain.nn.primitives.skip import SkipConnection


@pytest.fixture
def basic_input() -> torch.Tensor:
    return torch.randn(2, 4, 16)


@pytest.fixture
def linear_skip_sum() -> SkipConnection:
    """SkipConnection wrapping Linear(4→8), sum mode."""
    return SkipConnection(nn.Linear(4, 8), how="sum", layer_type="linear")


@pytest.fixture
def linear_skip_concat() -> SkipConnection:
    """SkipConnection wrapping Linear(4→8), concat mode (output width = 16)."""
    return SkipConnection(nn.Linear(4, 8), how="concat", layer_type="linear")


@pytest.fixture
def skip_input() -> torch.Tensor:
    """Batch of 3 samples with 4 features."""
    return torch.randn(3, 4)
