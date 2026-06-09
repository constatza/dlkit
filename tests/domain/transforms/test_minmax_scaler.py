from __future__ import annotations

import pytest
import torch

from dlkit.domain.transforms.minmax import MinMaxScaler


@pytest.fixture
def data_2d() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(100, 10)


@pytest.fixture
def data_3d() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(64, 20, 5)


def test_minmax_scaler_fit_transform_inverse() -> None:
    x1 = torch.tensor([[0.0, 1.0], [2.0, 3.0]])

    t = MinMaxScaler(dim=0)
    t.fit(x1)
    assert t.fitted
    assert torch.allclose(t.min, torch.tensor([[0.0, 1.0]]))
    assert torch.allclose(t.max, torch.tensor([[2.0, 3.0]]))

    y = t.forward(x1)
    x_rec = t.inverse_transform(y)
    assert torch.allclose(x_rec, x1)


def test_minmax_scaler_incremental_fit_matches_full_fit() -> None:
    x1 = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    x2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_full = torch.cat([x1, x2], dim=0)

    full = MinMaxScaler(dim=0)
    full.fit(x_full)

    inc = MinMaxScaler(dim=0)
    inc.reset_fit_state()
    inc.update_fit(x1)
    inc.update_fit(x2)
    inc.finalize_fit()

    assert inc.fitted
    assert torch.allclose(inc.min, full.min)
    assert torch.allclose(inc.max, full.max)


def test_output_in_range(data_2d: torch.Tensor) -> None:
    scaler = MinMaxScaler(dim=0)
    scaler.fit(data_2d)
    out = scaler(data_2d)
    assert out.min().item() >= -1.0 - 1e-6
    assert out.max().item() <= 1.0 + 1e-6


def test_dim_tuple_global_for_3d(data_3d: torch.Tensor) -> None:
    scaler = MinMaxScaler(dim=(0, 1))
    scaler.fit(data_3d)
    assert scaler.min.shape == (1, 1, 5), f"expected (1,1,5), got {scaler.min.shape}"
    out = scaler(data_3d)
    assert out.min().item() >= -1.0 - 1e-6
    assert out.max().item() <= 1.0 + 1e-6


def test_dim_zero_per_position_for_3d(data_3d: torch.Tensor) -> None:
    scaler = MinMaxScaler(dim=0)
    scaler.fit(data_3d)
    assert scaler.min.shape == (1, 20, 5), f"expected (1,20,5), got {scaler.min.shape}"


def test_dim_not_mutated_after_fit(data_2d: torch.Tensor) -> None:
    scaler = MinMaxScaler(dim=0)
    scaler.fit(data_2d)
    assert scaler.dim == (0,)


def test_custom_interval_output_in_range(data_2d: torch.Tensor) -> None:
    scaler = MinMaxScaler(dim=0, low=0.0, high=1.0)
    scaler.fit(data_2d)
    out = scaler(data_2d)
    assert out.min().item() >= 0.0 - 1e-6
    assert out.max().item() <= 1.0 + 1e-6


def test_custom_interval_round_trip(data_2d: torch.Tensor) -> None:
    scaler = MinMaxScaler(dim=0, low=0.0, high=1.0)
    scaler.fit(data_2d)
    assert torch.allclose(scaler.inverse_transform(scaler(data_2d)), data_2d, atol=1e-5)


def test_default_interval_backward_compatible(data_2d: torch.Tensor) -> None:
    scaler = MinMaxScaler(dim=0)
    scaler.fit(data_2d)
    out = scaler(data_2d)
    assert out.min().item() >= -1.0 - 1e-6
    assert out.max().item() <= 1.0 + 1e-6


def test_invalid_interval_raises() -> None:
    with pytest.raises(ValueError, match="high"):
        MinMaxScaler(dim=0, low=1.0, high=0.0)

    with pytest.raises(ValueError, match="high"):
        MinMaxScaler(dim=0, low=0.5, high=0.5)
