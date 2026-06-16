from __future__ import annotations

import pytest
import torch
from torch import Tensor

from dlkit.domain.transforms.base import InvertibleTransform
from dlkit.domain.transforms.unsqueeze import Unsqueeze


@pytest.fixture
def data_2d() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(16, 8)


@pytest.fixture
def data_3d() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 10, 8)


def test_forward_shape_dim1(data_2d: Tensor) -> None:
    assert Unsqueeze(dim=1)(data_2d).shape == (16, 1, 8)


def test_forward_shape_dim0(data_2d: Tensor) -> None:
    assert Unsqueeze(dim=0)(data_2d).shape == (1, 16, 8)


def test_forward_shape_neg_dim(data_2d: Tensor) -> None:
    assert Unsqueeze(dim=-1)(data_2d).shape == (16, 8, 1)


def test_round_trip_dim1(data_2d: Tensor) -> None:
    t = Unsqueeze(dim=1)
    assert torch.allclose(t.inverse_transform(t(data_2d)), data_2d)


def test_round_trip_3d(data_3d: Tensor) -> None:
    t = Unsqueeze(dim=2)
    assert torch.allclose(t.inverse_transform(t(data_3d)), data_3d)


@pytest.mark.parametrize(
    ("in_shape", "dim", "expected"),
    [
        ((16, 8), 1, (16, 1, 8)),
        ((16, 8), 0, (1, 16, 8)),
        ((16, 8), -1, (16, 8, 1)),
        ((4, 10, 8), 2, (4, 10, 1, 8)),
        ((4, 10, 8), -2, (4, 10, 1, 8)),
    ],
)
def test_infer_output_shape(in_shape: tuple[int, ...], dim: int, expected: tuple[int, ...]) -> None:
    assert Unsqueeze(dim=dim).infer_output_shape(in_shape) == expected


def test_invertible_protocol() -> None:
    assert isinstance(Unsqueeze(dim=1), InvertibleTransform)
