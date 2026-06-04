"""Memory-bounded tests for the zarr dense array pack.

Each test verifies that the public API does not accumulate the full dataset
in RAM.  The memory limits are tight enough to catch any naive in-memory
accumulation while leaving room for Python / zarr overhead.

Total data size for the 1_000-sample 128×128 float32 pack:
    1000 × 128 × 128 × 4 bytes = 65.5 MB

Any implementation that loads or retains the whole dataset in memory will
exceed the per-test limits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlkit.infrastructure.io.packs import open_array_pack, write_array_pack


@pytest.mark.limit_memory("35 MB")
def test_streaming_write_is_bounded(tmp_path: Path, matrix_stream) -> None:
    """Writing 1_000 matrices one at a time must not accumulate >35 MB peak RAM.

    Total data is 65 MB dense — any per-sample accumulation would exceed the
    limit long before all samples are written.

    Args:
        tmp_path: pytest temporary directory.
        matrix_stream: Generator of 1_000 random (128, 128) float32 arrays.
    """
    with write_array_pack(tmp_path / "pack", size=(128, 128)) as w:
        for matrix in matrix_stream:
            w.write_sample(matrix)


@pytest.mark.limit_memory("50 MB")
def test_batch_read_does_not_load_dataset(pack_1k: Path) -> None:
    """Reading 4 samples from a 1_000-sample pack must not load the full 65 MB.

    Args:
        pack_1k: Session-scoped 1_000-sample zarr dense pack.
    """
    reader = open_array_pack(pack_1k)
    batch = reader[[0, 100, 500, 999]]
    assert batch.shape == (4, 128, 128)


@pytest.mark.limit_memory("80 MB")
def test_dataloader_epoch_bounded(pack_1k: Path, tmp_path: Path) -> None:
    """Full DataLoader epoch with num_workers=0 must stay under 80 MB peak.

    Uses ``collate_tensordict`` because the default collate cannot handle
    ``TensorDict`` items (PEP 479 / ``StopIteration`` propagation).

    Args:
        pack_1k: Session-scoped 1_000-sample zarr dense pack.
        tmp_path: pytest temporary directory.
    """
    from torch.utils.data import DataLoader

    from dlkit.engine.data.datasets import FlexibleDataset
    from dlkit.engine.data.datasets.flexible import collate_tensordict
    from dlkit.infrastructure.config.entry_factories import Target
    from dlkit.infrastructure.config.entry_types import PathFeature

    n = 1_000
    target_path = tmp_path / "targets.npy"
    np.save(target_path, np.zeros((n, 1), dtype=np.float32))

    dataset = FlexibleDataset(
        features=[PathFeature(name="mat", path=pack_1k)],
        targets=[Target(name="y", path=target_path)],
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_tensordict,
    )
    for _batch in loader:
        pass
