from __future__ import annotations

import pytest
import torch
from torch import Tensor

from dlkit.domain.transforms.log_transform import LogTransform


@pytest.fixture
def positive_data() -> Tensor:
    torch.manual_seed(0)
    return torch.rand(16, 8) * 5.0 + 0.1


@pytest.fixture
def positive_data_3d() -> Tensor:
    torch.manual_seed(0)
    return torch.rand(4, 10, 8) * 5.0 + 0.1


def test_round_trip(positive_data: Tensor) -> None:
    t = LogTransform(shift=1.0)
    assert torch.allclose(t.inverse_transform(t(positive_data)), positive_data, atol=1e-5)


def test_round_trip_custom_shift(positive_data: Tensor) -> None:
    t = LogTransform(shift=0.5)
    assert torch.allclose(t.inverse_transform(t(positive_data)), positive_data, atol=1e-5)


def test_round_trip_with_indices(positive_data: Tensor) -> None:
    t = LogTransform(shift=1.0, indices=[0, 3, 7], index_dim=-1)
    assert torch.allclose(t.inverse_transform(t(positive_data)), positive_data, atol=1e-5)


def test_selected_features_transformed(positive_data: Tensor) -> None:
    t = LogTransform(shift=1.0, indices=[0, 2], index_dim=-1)
    y = t(positive_data)
    expected = torch.log(positive_data[:, 0] + 1.0)
    assert torch.allclose(y[:, 0], expected, atol=1e-6)


def test_unselected_features_unchanged(positive_data: Tensor) -> None:
    t = LogTransform(shift=1.0, indices=[0], index_dim=-1)
    y = t(positive_data)
    assert torch.allclose(y[:, 1:], positive_data[:, 1:])


def test_shape_preserved(positive_data: Tensor) -> None:
    assert LogTransform()(positive_data).shape == positive_data.shape


def test_3d_round_trip(positive_data_3d: Tensor) -> None:
    t = LogTransform(shift=1.0, indices=[0, 2], index_dim=-1)
    assert torch.allclose(t.inverse_transform(t(positive_data_3d)), positive_data_3d, atol=1e-5)


def test_invalid_shift_raises() -> None:
    with pytest.raises(ValueError, match="shift"):
        LogTransform(shift=0.0)

    with pytest.raises(ValueError, match="shift"):
        LogTransform(shift=-1.0)
