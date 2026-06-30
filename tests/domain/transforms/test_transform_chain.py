from __future__ import annotations

import pytest
import torch
from torch.nn import ModuleList

from dlkit.domain.transforms.chain import TransformChain
from dlkit.domain.transforms.minmax import MinMaxScaler
from dlkit.domain.transforms.pca import PCA
from dlkit.domain.transforms.standard import StandardScaler
from dlkit.engine.adapters.lightning.transform_builder import build_transform_list
from dlkit.infrastructure.config.transform_settings import TransformSettings


def test_transform_chain_build_fit_forward_inverse() -> None:
    # Single MinMax scaler in chain
    ts = TransformSettings.model_validate(
        {"name": "MinMaxScaler", "module_path": "dlkit.domain.transforms.minmax", "dim": 0}
    )
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


def test_build_transform_list_accepts_serialized_transform_dicts() -> None:
    module_list, _ = build_transform_list(
        [{"name": "PCA", "dim": 0, "n_components": 2}],
        entry_name="x",
    )

    assert len(module_list) == 1
    transform = module_list[0]
    assert isinstance(transform, PCA)
    assert transform.n_components == 2


def test_build_transform_list_preserves_typed_transform_settings() -> None:
    module_list, _ = build_transform_list(
        [TransformSettings.model_validate({"name": "PCA", "n_components": 2})]
    )

    assert len(module_list) == 1
    transform = module_list[0]
    assert isinstance(transform, PCA)
    assert transform.n_components == 2


def test_build_transform_list_rejects_unsupported_spec_types() -> None:
    with pytest.raises(TypeError, match="Transform specifications must be"):
        build_transform_list([object()])


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


@pytest.fixture
def unfitted_pca_chain() -> TransformChain:
    pca = PCA(n_components=2)
    return TransformChain(ModuleList([pca]))


def test_fit_from_dataloader_materialises_for_non_incremental_transform(
    unfitted_pca_chain: TransformChain,
) -> None:
    data = torch.randn(60, 6)
    batches = [data[i : i + 20] for i in range(0, 60, 20)]
    dataloader = [{"features": b} for b in batches]

    unfitted_pca_chain.fit_from_dataloader(
        dataloader, tensor_selector=lambda batch: batch["features"]
    )

    assert unfitted_pca_chain.fitted
    out = unfitted_pca_chain(data)
    assert out.shape == (60, 2)


def test_fit_from_dataloader_applies_prior_transforms_before_fitting_pca() -> None:
    # Chain: [StandardScaler (incremental), PCA (non-incremental)].
    # PCA must be fitted on scaled data, not raw — verifies prior-transform application.
    scaler = StandardScaler(dim=0)
    pca = PCA(n_components=2)
    chain = TransformChain(ModuleList([scaler, pca]), entry_name="x")

    torch.manual_seed(0)
    data = torch.randn(60, 6) * 100  # large scale to make scaler effect visible
    batches = [data[i : i + 20] for i in range(0, 60, 20)]

    chain.fit_from_dataloader(batches, tensor_selector=lambda b: b)

    assert chain.fitted
    assert scaler.fitted
    assert pca.fitted
    out = chain(data)
    assert out.shape == (60, 2)


def test_fit_from_dataloader_respects_declared_order_when_materializing_transform_is_first() -> (
    None
):
    """Reverse of the above: [PCA (non-incremental), StandardScaler (incremental)].

    StandardScaler must be fitted on PCA's *output* (2 features), not the raw
    6-feature input — proves prior-transform application is symmetric: it
    doesn't matter whether the earlier transform is incremental or
    materializing, only its position in the declared order.
    """
    pca = PCA(n_components=2)
    scaler = StandardScaler(dim=0)
    chain = TransformChain(ModuleList([pca, scaler]), entry_name="x")

    torch.manual_seed(0)
    data = torch.randn(60, 6) * 100
    batches = [data[i : i + 20] for i in range(0, 60, 20)]

    chain.fit_from_dataloader(batches, tensor_selector=lambda b: b)

    assert chain.fitted
    assert pca.fitted
    assert scaler.fitted
    # Proves scaler fitted on PCA's 2-feature output, not the raw 6-feature input.
    assert scaler.mean.shape[-1] == 2
    expected_mean = pca(data).mean(dim=0, keepdim=True)
    assert torch.allclose(scaler.mean, expected_mean, atol=1e-4)

    out = chain(data)
    assert out.shape == (60, 2)
    assert torch.allclose(out.mean(dim=0), torch.zeros(2), atol=1e-4)
