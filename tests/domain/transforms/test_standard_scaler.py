from __future__ import annotations

import pytest
import torch

from dlkit.domain.transforms.standard import StandardScaler


@pytest.fixture
def data_2d() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(100, 10)


@pytest.fixture
def data_3d() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(64, 20, 5)


def test_standard_scaler_fit_transform_inverse() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    scaler = StandardScaler(dim=0)
    scaler.fit(x)

    y = scaler.forward(x)
    x_rec = scaler.inverse_transform(y)

    assert scaler.fitted
    assert torch.allclose(x_rec, x, atol=1e-6)


def test_standard_scaler_incremental_fit_matches_full_fit() -> None:
    torch.manual_seed(7)
    x = torch.randn(40, 8)
    chunks = (x[:10], x[10:22], x[22:])

    full = StandardScaler(dim=0)
    full.fit(x)

    inc = StandardScaler(dim=0)
    inc.reset_fit_state()
    for chunk in chunks:
        inc.update_fit(chunk)
    inc.finalize_fit()

    assert inc.fitted
    assert torch.allclose(inc.mean, full.mean, atol=1e-6, rtol=1e-5)
    assert torch.allclose(inc.std, full.std, atol=1e-6, rtol=1e-5)


def test_output_zero_mean_unit_std(data_2d: torch.Tensor) -> None:
    scaler = StandardScaler(dim=0)
    scaler.fit(data_2d)
    out = scaler(data_2d)
    assert torch.allclose(out.mean(dim=0), torch.zeros(10), atol=1e-5)
    # scaler uses population std (unbiased=False), so verify with correction=0
    assert torch.allclose(out.std(dim=0, correction=0), torch.ones(10), atol=1e-3)


def test_dim_not_mutated_after_fit(data_2d: torch.Tensor) -> None:
    scaler = StandardScaler(dim=0)
    original_dim = scaler.dim
    scaler.fit(data_2d)
    assert scaler.dim == original_dim


def test_dim_tuple_global_for_3d(data_3d: torch.Tensor) -> None:
    scaler = StandardScaler(dim=[0, 1])
    scaler.fit(data_3d)
    assert scaler.mean.shape == (1, 1, 5), f"expected (1,1,5), got {scaler.mean.shape}"
    out = scaler(data_3d)
    assert torch.allclose(out.mean(dim=(0, 1)), torch.zeros(5), atol=1e-4)
