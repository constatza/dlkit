from __future__ import annotations

import pytest
import torch
from torch import Tensor

from dlkit.domain.transforms.base import InvertibleTransform, PartialTransform


class _Double(PartialTransform):
    """Minimal concrete PartialTransform for testing."""

    def _compute(self, x: Tensor) -> Tensor:
        return x * 2.0

    def _inverse_compute(self, x: Tensor) -> Tensor:
        return x / 2.0


@pytest.fixture
def data_2d() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(16, 8)


@pytest.fixture
def data_3d() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 10, 8)


def test_no_indices_applies_to_all(data_2d: Tensor) -> None:
    t = _Double()
    assert torch.allclose(t(data_2d), data_2d * 2.0)


def test_round_trip_no_indices(data_2d: Tensor) -> None:
    t = _Double()
    assert torch.allclose(t.inverse_transform(t(data_2d)), data_2d)


def test_round_trip_with_indices(data_2d: Tensor) -> None:
    t = _Double(indices=[0, 3, 7], index_dim=-1)
    assert torch.allclose(t.inverse_transform(t(data_2d)), data_2d)


def test_selected_features_transformed(data_2d: Tensor) -> None:
    t = _Double(indices=[1, 4], index_dim=-1)
    y = t(data_2d)
    assert torch.allclose(y[:, 1], data_2d[:, 1] * 2.0)
    assert torch.allclose(y[:, 4], data_2d[:, 4] * 2.0)


def test_unselected_features_unchanged(data_2d: Tensor) -> None:
    t = _Double(indices=[1, 4], index_dim=-1)
    y = t(data_2d)
    for i in [0, 2, 3, 5, 6, 7]:
        assert torch.allclose(y[:, i], data_2d[:, i])


def test_shape_preserved_2d(data_2d: Tensor) -> None:
    assert _Double(indices=[0, 2])(data_2d).shape == data_2d.shape


def test_shape_preserved_3d(data_3d: Tensor) -> None:
    assert _Double(indices=[0, 2])(data_3d).shape == data_3d.shape


def test_3d_indices_on_last_dim(data_3d: Tensor) -> None:
    t = _Double(indices=[0, 1], index_dim=-1)
    y = t(data_3d)
    assert torch.allclose(y[..., 0], data_3d[..., 0] * 2.0)
    assert torch.allclose(y[..., 2], data_3d[..., 2])


def test_satisfies_invertible_protocol() -> None:
    assert isinstance(_Double(), InvertibleTransform)


def test_fitted_starts_false() -> None:
    t = _Double()
    assert t.fitted is False
