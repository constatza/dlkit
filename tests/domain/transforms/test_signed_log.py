from __future__ import annotations

import pytest
import torch
from torch import Tensor

from dlkit.domain.transforms.signed_log import SignedLogTransform


@pytest.fixture
def mixed_data() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(16, 8) * 3.0


@pytest.fixture
def mixed_data_3d() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 10, 8) * 3.0


def test_round_trip(mixed_data: Tensor) -> None:
    t = SignedLogTransform(shift=1.0)
    assert torch.allclose(t.inverse_transform(t(mixed_data)), mixed_data, atol=1e-5)


def test_round_trip_with_indices(mixed_data: Tensor) -> None:
    t = SignedLogTransform(shift=1.0, indices=[0, 3, 7], index_dim=-1)
    assert torch.allclose(t.inverse_transform(t(mixed_data)), mixed_data, atol=1e-5)


def test_sign_preserved(mixed_data: Tensor) -> None:
    t = SignedLogTransform(shift=1.0)
    y = t(mixed_data)
    assert torch.all(torch.sign(y) == torch.sign(mixed_data))


def test_handles_negative_values() -> None:
    x = torch.tensor([[-3.0, 0.0, 3.0]])
    t = SignedLogTransform(shift=1.0)
    y = t(x)
    assert torch.allclose(t.inverse_transform(y), x, atol=1e-5)


def test_shape_preserved(mixed_data: Tensor) -> None:
    assert SignedLogTransform()(mixed_data).shape == mixed_data.shape


def test_3d_round_trip(mixed_data_3d: Tensor) -> None:
    t = SignedLogTransform(shift=1.0, indices=[1, 3], index_dim=-1)
    assert torch.allclose(t.inverse_transform(t(mixed_data_3d)), mixed_data_3d, atol=1e-5)


def test_invalid_shift_raises() -> None:
    with pytest.raises(ValueError, match="shift"):
        SignedLogTransform(shift=0.0)
