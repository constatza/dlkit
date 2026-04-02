from __future__ import annotations

import pytest
import torch
from torch.nn import ModuleList

from dlkit.domain.transforms.chain import TransformChain
from dlkit.domain.transforms.minmax import MinMaxScaler
from dlkit.domain.transforms.pca import PCA
from dlkit.domain.transforms.standard import StandardScaler
from dlkit.tools.config.transform_settings import TransformSettings


def test_transform_chain_build_fit_forward_inverse() -> None:
    # Single MinMax scaler in chain
    from dlkit.runtime.workflows.factories.component_builders import build_transform_list

    ts = TransformSettings(name="MinMaxScaler", module_path="dlkit.domain.transforms.minmax", dim=0)
    # No shape_spec needed - transforms will infer from data
    module_list, _ = build_transform_list([ts])
    chain = TransformChain(module_list)

    x = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    chain.fit(x)
    y = chain.forward(x)
    x_rec = chain.inverse_transform(y)

    assert chain.fitted
    transformed_shape = chain.transformed_shape
    assert transformed_shape is not None
    assert tuple(transformed_shape) == tuple(x.shape)
    assert torch.allclose(x_rec, x)

    inv = chain.inverse()
    assert isinstance(inv, TransformChain)


def test_transform_chain_streaming_fit_multi_transform() -> None:
    chain = TransformChain(
        ModuleList([MinMaxScaler(dim=0), StandardScaler(dim=0)]),
        entry_name="x",
    )
    dataloader = [
        torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    ]

    chain.fit_from_dataloader(dataloader, tensor_selector=lambda batch: batch)

    assert chain.fitted
    y = chain.forward(dataloader[0])
    assert y.shape == dataloader[0].shape


def test_transform_chain_streaming_fit_fails_for_unfitted_non_incremental() -> None:
    chain = TransformChain(ModuleList([PCA(n_components=1)]), entry_name="x")
    dataloader = [torch.randn(4, 3), torch.randn(4, 3)]

    with pytest.raises(TypeError, match="TODO: incremental PCA"):
        chain.fit_from_dataloader(dataloader, tensor_selector=lambda batch: batch)


def test_transform_chain_streaming_fit_allows_prefitted_pca() -> None:
    pca = PCA(n_components=1)
    full = torch.randn(8, 3)
    pca.fit(full)

    chain = TransformChain(ModuleList([pca]), entry_name="x")
    dataloader = [full[:4], full[4:]]

    chain.fit_from_dataloader(dataloader, tensor_selector=lambda batch: batch)

    assert chain.fitted
    y = chain.forward(dataloader[0])
    assert y.shape[-1] == 1
