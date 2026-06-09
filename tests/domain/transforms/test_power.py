from __future__ import annotations

import pytest
import torch
from torch import Tensor

from dlkit.domain.transforms.power import PowerTransform


@pytest.fixture
def positive_data() -> Tensor:
    torch.manual_seed(0)
    return torch.rand(16, 8) * 4.0 + 0.1


@pytest.fixture
def positive_data_3d() -> Tensor:
    torch.manual_seed(0)
    return torch.rand(4, 10, 8) * 4.0 + 0.1


def test_round_trip_sqrt(positive_data: Tensor) -> None:
    t = PowerTransform(exponent=0.5)
    assert torch.allclose(t.inverse_transform(t(positive_data)), positive_data, atol=1e-5)


def test_round_trip_square(positive_data: Tensor) -> None:
    t = PowerTransform(exponent=2.0)
    assert torch.allclose(t.inverse_transform(t(positive_data)), positive_data, atol=1e-4)


def test_round_trip_with_indices(positive_data: Tensor) -> None:
    t = PowerTransform(exponent=0.5, indices=[0, 2, 5], index_dim=-1)
    assert torch.allclose(t.inverse_transform(t(positive_data)), positive_data, atol=1e-5)


def test_sqrt_values(positive_data: Tensor) -> None:
    t = PowerTransform(exponent=0.5)
    assert torch.allclose(t(positive_data), torch.sqrt(positive_data), atol=1e-6)


def test_unselected_unchanged(positive_data: Tensor) -> None:
    t = PowerTransform(exponent=0.5, indices=[0], index_dim=-1)
    y = t(positive_data)
    assert torch.allclose(y[:, 1:], positive_data[:, 1:])


def test_shape_preserved(positive_data: Tensor) -> None:
    assert PowerTransform(exponent=2.0)(positive_data).shape == positive_data.shape


def test_3d_round_trip(positive_data_3d: Tensor) -> None:
    t = PowerTransform(exponent=0.5, indices=[0, 3], index_dim=-1)
    assert torch.allclose(t.inverse_transform(t(positive_data_3d)), positive_data_3d, atol=1e-5)


def test_zero_exponent_raises() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        PowerTransform(exponent=0)
