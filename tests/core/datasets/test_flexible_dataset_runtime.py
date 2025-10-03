from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dlkit.core.datasets.flexible import FlexibleDataset


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
        features=[{"name": "x1", "path": x1_path}, {"name": "x2", "path": x2_path}],
        targets=[{"name": "y", "path": y_path}],
    )

    assert len(ds) == 5
    sample0 = ds[0]
    assert set(sample0.keys()) == {"x1", "x2", "y"}
    assert isinstance(sample0["x1"], torch.Tensor)
    assert tuple(sample0["x1"].shape) == (2,)
    assert tuple(sample0["x2"].shape) == (3,)
    assert tuple(sample0["y"].shape) == (1,)
