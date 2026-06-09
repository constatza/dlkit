from __future__ import annotations

import pytest
import torch
from torch import Tensor

from dlkit.domain.transforms.logit_transform import LogitTransform


@pytest.fixture
def probability_data() -> Tensor:
    torch.manual_seed(0)
    return torch.rand(16, 8) * 0.9 + 0.05  # values in (0.05, 0.95)


@pytest.fixture
def probability_data_3d() -> Tensor:
    torch.manual_seed(0)
    return torch.rand(4, 10, 8) * 0.9 + 0.05


def test_round_trip(probability_data: Tensor) -> None:
    t = LogitTransform()
    assert torch.allclose(t.inverse_transform(t(probability_data)), probability_data, atol=1e-5)


def test_round_trip_with_indices(probability_data: Tensor) -> None:
    t = LogitTransform(indices=[0, 3], index_dim=-1)
    assert torch.allclose(t.inverse_transform(t(probability_data)), probability_data, atol=1e-5)


def test_clamps_boundary_values() -> None:
    t = LogitTransform(eps=1e-6)
    x = torch.tensor([[0.0, 0.5, 1.0]])
    y = t(x)
    assert torch.isfinite(y).all()


def test_inverse_is_sigmoid(probability_data: Tensor) -> None:
    t = LogitTransform()
    y = t(probability_data)
    assert torch.allclose(t.inverse_transform(y), torch.sigmoid(y), atol=1e-6)


def test_unselected_unchanged(probability_data: Tensor) -> None:
    t = LogitTransform(indices=[0], index_dim=-1)
    y = t(probability_data)
    assert torch.allclose(y[:, 1:], probability_data[:, 1:])


def test_shape_preserved(probability_data: Tensor) -> None:
    assert LogitTransform()(probability_data).shape == probability_data.shape


def test_3d_round_trip(probability_data_3d: Tensor) -> None:
    t = LogitTransform(indices=[0, 2], index_dim=-1)
    assert torch.allclose(
        t.inverse_transform(t(probability_data_3d)), probability_data_3d, atol=1e-5
    )


def test_invalid_eps_raises() -> None:
    with pytest.raises(ValueError, match="eps"):
        LogitTransform(eps=0.0)

    with pytest.raises(ValueError, match="eps"):
        LogitTransform(eps=0.6)
