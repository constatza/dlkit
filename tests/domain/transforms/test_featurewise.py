from __future__ import annotations

import pytest
import torch
from torch import Tensor

from dlkit.domain.transforms.affine import AffineTransform
from dlkit.domain.transforms.featurewise import FeatureWise
from dlkit.domain.transforms.minmax import MinMaxScaler
from dlkit.domain.transforms.standard import StandardScaler


@pytest.fixture
def data() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(32, 10)


@pytest.fixture
def data_3d() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 20, 10)


def test_round_trip_with_invertible(data: Tensor) -> None:
    t = FeatureWise(MinMaxScaler(dim=0), indices=[0, 2, 5])
    t.fit(data)
    assert torch.allclose(t.inverse_transform(t(data)), data, atol=1e-5)


def test_selected_features_transformed(data: Tensor) -> None:
    inner = MinMaxScaler(dim=0)
    t = FeatureWise(inner, indices=[1, 3])
    t.fit(data)
    y = t(data)
    # unselected features pass through unchanged
    for i in [0, 2, 4, 5, 6, 7, 8, 9]:
        assert torch.allclose(y[:, i], data[:, i])


def test_fit_delegates_to_inner(data: Tensor) -> None:
    inner = StandardScaler(dim=0)
    t = FeatureWise(inner, indices=[0, 1])
    assert not t.fitted
    t.fit(data)
    assert t.fitted
    assert inner.fitted


def test_incremental_fit(data: Tensor) -> None:
    half = data.shape[0] // 2
    x1, x2 = data[:half], data[half:]

    full = FeatureWise(MinMaxScaler(dim=0), indices=[0, 2])
    full.fit(data)

    inc = FeatureWise(MinMaxScaler(dim=0), indices=[0, 2])
    inc.reset_fit_state()
    inc.update_fit(x1)
    inc.update_fit(x2)
    inc.finalize_fit()

    assert torch.allclose(full(data), inc(data), atol=1e-5)


def test_inverse_raises_for_non_invertible() -> None:

    from dlkit.domain.transforms.base import Transform

    class _NoInverse(Transform):
        def forward(self, x: Tensor) -> Tensor:
            return x

    t = FeatureWise(_NoInverse(), indices=[0])
    with pytest.raises(TypeError, match="inverse_transform"):
        t.inverse_transform(torch.randn(4, 4))


def test_unfittable_inner_always_fitted() -> None:
    t = FeatureWise(AffineTransform(scale=2.0), indices=[0])
    assert t.fitted


def test_3d_round_trip(data_3d: Tensor) -> None:
    t = FeatureWise(MinMaxScaler(dim=0), indices=[0, 3, 7], index_dim=-1)
    t.fit(data_3d)
    assert torch.allclose(t.inverse_transform(t(data_3d)), data_3d, atol=1e-5)


def test_shape_preserved(data: Tensor) -> None:
    t = FeatureWise(AffineTransform(scale=2.0), indices=[0, 1])
    assert t(data).shape == data.shape
