from __future__ import annotations

import torch

from dlkit.domain.transforms.minmax import MinMaxScaler


def test_minmax_scaler_fit_transform_inverse() -> None:
    x1 = torch.tensor([[0.0, 1.0], [2.0, 3.0]])

    t = MinMaxScaler(dim=0)
    t.fit(x1)
    assert t.fitted
    # min should be [0,1], max should be [2,3]
    assert torch.allclose(t.min, torch.tensor([[0.0, 1.0]]))
    assert torch.allclose(t.max, torch.tensor([[2.0, 3.0]]))

    y = t.forward(x1)
    # Inverse should approximately recover input
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
