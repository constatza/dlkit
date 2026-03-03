"""Sparse feature integration tests for FlexibleDataset."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dlkit.core.datasets.flexible import FlexibleDataset, collate_tensordict
from dlkit.tools.config.data_entries import Feature, SparseFeature, Target


def test_flexible_dataset_sparse_collation(sparse_collation_pack: dict[str, Any]) -> None:
    pack_path = sparse_collation_pack["path"]
    matrices = sparse_collation_pack["matrices"]
    n_samples = len(matrices)
    dim = matrices[0].shape[0]
    x = torch.randn(n_samples, 2)
    y = torch.randn(n_samples, 1)

    dataset = FlexibleDataset(
        features=[
            Feature(name="x", value=x),
            SparseFeature(name="matrix", path=pack_path, model_input=False, loss_input="matrix"),
        ],
        targets=[Target(name="y", value=y)],
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_tensordict)
    batch = next(iter(loader))
    sparse_batch = batch["features"]["matrix"]

    assert sparse_batch.shape == (2, dim, dim)
    assert sparse_batch.is_sparse
    assert torch.allclose(
        sparse_batch[0].to_dense(),
        torch.from_numpy(matrices[0]).to(dtype=sparse_batch.dtype),
    )
    assert torch.allclose(
        sparse_batch[1].to_dense(),
        torch.from_numpy(matrices[1]).to(dtype=sparse_batch.dtype),
    )


def test_dataloader_uses_stacked_sparse_builder_for_non_shared_pack(
    sparse_collation_pack: dict[str, Any],
) -> None:
    """DataLoader batch path should invoke stacked sparse builder once per feature."""
    pack_path = sparse_collation_pack["path"]
    n_samples = len(sparse_collation_pack["matrices"])

    from dlkit.tools.io.sparse._coo_pack import CooPackReader

    original_build_stacked = CooPackReader.build_torch_sparse_stacked
    with patch(
        "dlkit.tools.io.sparse._coo_pack.CooPackReader.build_torch_sparse_stacked",
        autospec=True,
        wraps=original_build_stacked,
    ) as mocked_build_stacked:
        dataset = FlexibleDataset(
            features=[
                Feature(name="x", value=torch.randn(n_samples, 2)),
                SparseFeature(name="matrix", path=pack_path, model_input=False, loss_input="matrix"),
            ],
            targets=[Target(name="y", value=torch.randn(n_samples, 1))],
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_tensordict)
        _ = next(iter(loader))

    assert mocked_build_stacked.call_count == 1


def test_collate_tensordict_passthrough_for_prebatched_getitems(
    sparse_collation_pack: dict[str, Any],
) -> None:
    """collate_tensordict should pass through TensorDict batches from __getitems__."""
    pack_path = sparse_collation_pack["path"]
    n_samples = len(sparse_collation_pack["matrices"])
    dataset = FlexibleDataset(
        features=[
            Feature(name="x", value=torch.randn(n_samples, 2)),
            SparseFeature(name="matrix", path=pack_path, model_input=False, loss_input="matrix"),
        ],
        targets=[Target(name="y", value=torch.randn(n_samples, 1))],
    )

    prebatched = dataset.__getitems__([0, 1])
    collated = collate_tensordict(prebatched)
    assert collated is prebatched


def test_flexible_dataset_shared_sparse_pack_broadcasts(
    sparse_shared_pack: dict[str, Any],
) -> None:
    n_samples = 3
    pack_path = sparse_shared_pack["path"]
    shared = sparse_shared_pack["matrix"]

    dataset = FlexibleDataset(
        features=[
            Feature(name="x", value=torch.randn(n_samples, 2)),
            SparseFeature(name="matrix", path=pack_path, model_input=False, loss_input="matrix"),
        ],
        targets=[Target(name="y", value=torch.randn(n_samples, 1))],
    )

    for idx in range(n_samples):
        matrix = dataset[idx]["features"]["matrix"]
        assert matrix.is_sparse
        assert torch.allclose(matrix.to_dense(), torch.from_numpy(shared).to(dtype=matrix.dtype))


def test_shared_sparse_pack_reads_sparse_on_demand_per_item(
    sparse_shared_pack: dict[str, Any],
) -> None:
    n_samples = 4
    pack_path = sparse_shared_pack["path"]
    from dlkit.tools.io.sparse._coo_pack import CooPackReader

    original_build_sparse = CooPackReader.build_torch_sparse
    with patch(
        "dlkit.tools.io.sparse._coo_pack.CooPackReader.build_torch_sparse",
        autospec=True,
        wraps=original_build_sparse,
    ) as mocked_build_sparse:
        dataset = FlexibleDataset(
            features=[
                Feature(name="x", value=torch.randn(n_samples, 2)),
                SparseFeature(name="matrix", path=pack_path, model_input=False, loss_input="matrix"),
            ],
            targets=[Target(name="y", value=torch.randn(n_samples, 1))],
        )
        # Sparse tensors are no longer pre-materialized at init time.
        assert mocked_build_sparse.call_count == 0
        _ = dataset[0]["features"]["matrix"]
        _ = dataset[1]["features"]["matrix"]

    assert mocked_build_sparse.call_count == 2


def test_flexible_dataset_sparse_feature_denormalize_applies_scale(
    sparse_scaled_pack: dict[str, Any],
) -> None:
    pack_path = sparse_scaled_pack["path"]
    n_samples = len(sparse_scaled_pack["matrices"])

    base_dataset = FlexibleDataset(
        features=[
            Feature(name="x", value=torch.randn(n_samples, 2)),
            SparseFeature(name="matrix", path=pack_path, model_input=False, loss_input="matrix"),
        ],
        targets=[Target(name="y", value=torch.randn(n_samples, 1))],
    )
    denorm_dataset = FlexibleDataset(
        features=[
            Feature(name="x", value=torch.randn(n_samples, 2)),
            SparseFeature(
                name="matrix",
                path=pack_path,
                model_input=False,
                loss_input="matrix",
                denormalize=True,
            ),
        ],
        targets=[Target(name="y", value=torch.randn(n_samples, 1))],
    )

    base = base_dataset[0]["features"]["matrix"].to_dense()
    denorm = denorm_dataset[0]["features"]["matrix"].to_dense()
    assert torch.allclose(denorm, base * sparse_scaled_pack["scale"])


def test_flexible_dataset_auto_detects_sparse_pack_from_path_feature(
    sparse_path_feature_pack: dict[str, Any],
) -> None:
    """Path-based features pointing at sparse packs should auto-open readers."""
    pack_path = sparse_path_feature_pack["path"]
    matrices = sparse_path_feature_pack["matrices"]
    n_samples = len(matrices)

    dataset = FlexibleDataset(
        features=[
            Feature(name="x", value=torch.randn(n_samples, 2)),
            Feature(name="matrix", path=pack_path, model_input=False, loss_input="matrix"),
        ],
        targets=[Target(name="y", value=torch.randn(n_samples, 1))],
    )

    for idx in range(n_samples):
        matrix = dataset[idx]["features"]["matrix"]
        assert matrix.is_sparse
        assert torch.allclose(
            matrix.to_dense(), torch.from_numpy(matrices[idx]).to(dtype=matrix.dtype)
        )
