from __future__ import annotations

import torch

from dlkit.domain.transforms.standard import StandardScaler


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
