from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import torch
from tensordict import TensorDict

from dlkit.core.datasets.flexible import FlexibleDataset
from dlkit.tools.config.data_entries import Feature, Target


def test_flexible_dataset_runtime_loads_and_indexes(tmp_path: Path):
    # Create simple arrays
    X1 = np.arange(10, dtype=np.float32).reshape(5, 2)
    X2 = np.ones((5, 3), dtype=np.float32)
    Y = (np.arange(5, dtype=np.float32) * 2).reshape(5, 1)

    x1_path = tmp_path / "X1.npy"
    x2_path = tmp_path / "X2.npy"
    y_path = tmp_path / "Y.npy"

    np.save(x1_path, X1)
    np.save(x2_path, X2)
    np.save(y_path, Y)

    ds = FlexibleDataset(
        features=[Feature(name="x1", path=x1_path), Feature(name="x2", path=x2_path)],
        targets=[Target(name="y", path=y_path)],
    )

    assert len(ds) == 5
    sample0 = ds[0]
    assert isinstance(sample0, TensorDict)
    features = cast("TensorDict", sample0["features"])
    targets = cast("TensorDict", sample0["targets"])
    assert len(features.keys()) == 2 and len(targets.keys()) == 1
    assert isinstance(sample0["features", "x1"], torch.Tensor)
    assert tuple(sample0["features", "x1"].shape) == (2,)
    assert tuple(sample0["features", "x2"].shape) == (3,)
    assert tuple(sample0["targets", "y"].shape) == (1,)
