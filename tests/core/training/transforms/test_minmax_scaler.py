from __future__ import annotations

import torch

from dlkit.core.training.transforms.minmax import MinMaxScaler


def test_minmax_scaler_fit_transform_inverse() -> None:
    x1 = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    x2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    t = MinMaxScaler(dim=0)

    # Fit twice to accumulate global min/max
    t.fit(x1)
    t.fit(x2)
    assert t.fitted
    # min should be [0,1], max should be [3,4]
    assert torch.allclose(t.min, torch.tensor([[0.0, 1.0]]))
    assert torch.allclose(t.max, torch.tensor([[3.0, 4.0]]))

    y = t.forward(x1)
    # Inverse should approximately recover input
    x_rec = t.inverse_transform(y)
    assert torch.allclose(x_rec, x1)
