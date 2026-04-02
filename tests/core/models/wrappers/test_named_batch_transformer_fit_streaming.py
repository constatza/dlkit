from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from torch.nn import ModuleList

from dlkit.domain.transforms.chain import TransformChain
from dlkit.domain.transforms.minmax import MinMaxScaler
from dlkit.domain.transforms.pca import PCA
from dlkit.runtime.adapters.lightning.components import NamedBatchTransformer


def _make_batch(x: torch.Tensor) -> TensorDict:
    batch_size = [x.shape[0]]
    return TensorDict(
        {
            "features": TensorDict({"x": x}, batch_size=batch_size),
            "targets": TensorDict({}, batch_size=batch_size),
        },
        batch_size=batch_size,
    )


def test_named_batch_transformer_fit_streaming_does_not_use_torch_cat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = TransformChain(ModuleList([MinMaxScaler(dim=0)]), entry_name="x")
    transformer = NamedBatchTransformer(feature_chains={"x": chain}, target_chains={})
    dataloader = [
        _make_batch(torch.tensor([[0.0, 1.0], [2.0, 3.0]])),
        _make_batch(torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
    ]

    def _cat_guard(*args, **kwargs):
        raise AssertionError("torch.cat should not be used by streaming fit path")

    monkeypatch.setattr("dlkit.runtime.adapters.lightning.components.torch.cat", _cat_guard)
    transformer.fit(dataloader)

    assert chain.fitted


def test_named_batch_transformer_fit_fails_for_unfitted_pca() -> None:
    chain = TransformChain(ModuleList([PCA(n_components=1)]), entry_name="x")
    transformer = NamedBatchTransformer(feature_chains={"x": chain}, target_chains={})
    dataloader = [_make_batch(torch.randn(4, 3))]

    with pytest.raises(TypeError, match="TODO: incremental PCA"):
        transformer.fit(dataloader)
