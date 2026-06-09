from __future__ import annotations

import pytest
import torch
from torch import Tensor

from dlkit.domain.transforms.tanh_transform import TanhTransform


@pytest.fixture
def data() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(16, 8)


@pytest.fixture
def data_3d() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 10, 8)


def test_round_trip(data: Tensor) -> None:
    t = TanhTransform()
    assert torch.allclose(t.inverse_transform(t(data)), data, atol=1e-5)


def test_round_trip_with_indices(data: Tensor) -> None:
    t = TanhTransform(indices=[0, 3, 7], index_dim=-1)
    assert torch.allclose(t.inverse_transform(t(data)), data, atol=1e-5)


def test_output_bounded(data: Tensor) -> None:
    t = TanhTransform()
    y = t(data)
    assert y.min().item() > -1.0
    assert y.max().item() < 1.0


def test_unselected_unchanged(data: Tensor) -> None:
    t = TanhTransform(indices=[0], index_dim=-1)
    y = t(data)
    assert torch.allclose(y[:, 1:], data[:, 1:])


def test_shape_preserved(data: Tensor) -> None:
    assert TanhTransform()(data).shape == data.shape


def test_3d_round_trip(data_3d: Tensor) -> None:
    t = TanhTransform(indices=[0, 2], index_dim=-1)
    assert torch.allclose(t.inverse_transform(t(data_3d)), data_3d, atol=1e-5)
