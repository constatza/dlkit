from __future__ import annotations

import torch

from dlkit.core.training.transforms.chain import TransformChain
from dlkit.tools.config.transform_settings import TransformSettings


def test_transform_chain_build_fit_forward_inverse() -> None:
    # Single MinMax scaler in chain
    ts = TransformSettings(
        name="MinMaxScaler", module_path="dlkit.core.training.transforms.minmax", dim=0
    )
    # No shape_spec needed - transforms will infer from data
    chain = TransformChain([ts])

    x = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    chain.fit(x)
    y = chain.forward(x)
    x_rec = chain.inverse_transform(y)

    assert chain.fitted
    assert tuple(chain.transformed_shape) == tuple(x.shape)
    assert torch.allclose(x_rec, x)

    inv = chain.inverse()
    assert isinstance(inv, TransformChain)
