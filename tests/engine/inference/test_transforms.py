from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dlkit.domain.transforms.chain import TransformChain
from dlkit.domain.transforms.pca import PCA
from dlkit.engine.inference.transforms import load_transforms_from_checkpoint
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.entry_types import NpyEntry
from dlkit.infrastructure.config.transform_settings import TransformSettings


def _make_serialized_feature_entry(tmp_path: Path) -> dict[str, object]:
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, np.zeros((4, 5), dtype=np.float32))
    entry = NpyEntry(
        name="x",
        path=feature_path,
        transforms=[TransformSettings.model_validate({"name": "PCA", "n_components": 2})],
        data_role=DataRole.FEATURE,
    )
    return entry.model_dump()


def _make_named_chain_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    feature_batch = torch.randn(12, 5)
    transform = PCA(n_components=2)
    transform.fit(feature_batch)

    state = {
        "_batch_transformer._feature_chains.x._fitted": torch.tensor(True),
    }
    for key, value in transform.state_dict().items():
        state[f"_batch_transformer._feature_chains.x.transforms.0.{key}"] = value
    return state


def test_load_transforms_from_checkpoint_accepts_serialized_transform_dicts(
    tmp_path: Path,
) -> None:
    checkpoint = {
        "state_dict": _make_named_chain_state(),
        "dlkit_metadata": {
            "entry_configs": [_make_serialized_feature_entry(tmp_path)],
        },
    }

    feature_transforms, target_transforms = load_transforms_from_checkpoint(checkpoint)

    assert "x" in feature_transforms
    assert target_transforms == {}

    chain = feature_transforms["x"]
    assert isinstance(chain, TransformChain)
    assert chain.fitted
    assert len(chain.transforms) == 1
    assert isinstance(chain.transforms[0], PCA)

    sample = torch.randn(3, 5)
    transformed = chain(sample)
    transformed_twice = chain(sample)

    assert transformed.shape == (3, 2)
    assert transformed_twice.shape == (3, 2)
