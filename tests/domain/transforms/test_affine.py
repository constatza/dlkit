from __future__ import annotations

import pytest
import torch
from torch import Tensor

from dlkit.domain.transforms.affine import AffineTransform


@pytest.fixture
def data() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(16, 8)


@pytest.fixture
def data_3d() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 10, 8)


def test_round_trip(data: Tensor) -> None:
    t = AffineTransform(scale=2.0, shift=-0.5)
    assert torch.allclose(t.inverse_transform(t(data)), data, atol=1e-5)


def test_round_trip_defaults(data: Tensor) -> None:
    t = AffineTransform()
    assert torch.allclose(t.inverse_transform(t(data)), data, atol=1e-6)


def test_round_trip_with_indices(data: Tensor) -> None:
    t = AffineTransform(scale=0.01, shift=3.0, indices=[0, 2, 5], index_dim=-1)
    assert torch.allclose(t.inverse_transform(t(data)), data, atol=1e-4)


def test_forward_values(data: Tensor) -> None:
    t = AffineTransform(scale=2.0, shift=1.0)
    assert torch.allclose(t(data), data * 2.0 + 1.0)


def test_unselected_unchanged(data: Tensor) -> None:
    t = AffineTransform(scale=3.0, shift=1.0, indices=[0], index_dim=-1)
    y = t(data)
    assert torch.allclose(y[:, 1:], data[:, 1:])


def test_shape_preserved(data: Tensor) -> None:
    assert AffineTransform(scale=0.5)(data).shape == data.shape


def test_3d_round_trip(data_3d: Tensor) -> None:
    t = AffineTransform(scale=0.1, shift=2.0, indices=[1, 4], index_dim=-1)
    assert torch.allclose(t.inverse_transform(t(data_3d)), data_3d, atol=1e-4)


def test_zero_scale_raises() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        AffineTransform(scale=0)
