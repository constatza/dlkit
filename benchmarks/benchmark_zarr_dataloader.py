"""Benchmark: zarr dense pack DataLoader throughput.

The key variable is chunk_size relative to access pattern:
  - chunk_size=1   → each sample is its own chunk → random access is O(1 sample)
  - chunk_size=128 → random access reads 128 samples to serve 1 → 128x amplification

Usage::

    uv run python benchmarks/benchmark_zarr_dataloader.py
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import zarr
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from dlkit.infrastructure.io.packs._manifest import ArrayPackManifest
from dlkit.infrastructure.io.packs._zarr_dense import (
    _DATA_ARRAY_NAME,
    ZARR_GROUP_NAME,
    ZarrDensePackReader,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_SAMPLES = 500  # 500 × 500×500 × 4 B = 0.5 GB uncompressed — fast to write
MATRIX_ROWS = 500
MATRIX_COLS = 500
DTYPE = np.float32
BATCH_SIZE = 32
N_WARMUP_BATCHES = 2
N_BENCH_BATCHES = 20

BYTES_PER_SAMPLE = MATRIX_ROWS * MATRIX_COLS * np.dtype(DTYPE).itemsize
TOTAL_GB = N_SAMPLES * BYTES_PER_SAMPLE / 1e9


# ---------------------------------------------------------------------------
# Pack writer
# ---------------------------------------------------------------------------


def _write_pack(path: Path, data: np.ndarray, chunk_size: int) -> None:
    """Write a zarr dense pack with no compression.

    Args:
        path: Destination directory.
        data: Source array of shape (N, rows, cols).
        chunk_size: Number of samples per zarr chunk.
    """
    path.mkdir(parents=True, exist_ok=True)
    n, rows, cols = data.shape
    group = zarr.open_group(str(path / ZARR_GROUP_NAME), mode="w")
    arr = group.create_array(
        _DATA_ARRAY_NAME,
        shape=data.shape,
        chunks=(chunk_size, rows, cols),
        dtype=data.dtype,
    )
    arr[:] = data
    manifest = ArrayPackManifest(
        n_samples=n,
        matrix_size=(rows, cols),
        dtype=np.dtype(DTYPE).name,
        chunk_size=chunk_size,
    )
    group.attrs["dlkit_manifest"] = manifest.model_dump(by_alias=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ZarrPackDataset(Dataset[Tensor]):
    """Dataset wrapper around a ZarrDensePackReader."""

    def __init__(self, reader: ZarrDensePackReader) -> None:
        self._reader = reader

    def __len__(self) -> int:
        return self._reader.n_samples

    def __getitem__(self, idx: int) -> Tensor:
        return self._reader[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _header(title: str) -> None:
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _measure(
    reader: ZarrDensePackReader,
    *,
    num_workers: int,
    shuffle: bool,
    label: str,
) -> None:
    """Run a DataLoader and print one result row.

    Args:
        reader: Pack reader to benchmark.
        num_workers: DataLoader worker count.
        shuffle: If True shuffle; False uses SequentialSampler.
        label: Row label.
    """
    ds = ZarrPackDataset(reader)
    sampler = None if shuffle else SequentialSampler(ds)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=False,
    )
    it = iter(loader)

    for _ in range(N_WARMUP_BATCHES):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)

    t0 = time.perf_counter()
    samples = 0
    batches = 0
    for _ in range(N_BENCH_BATCHES):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        samples += batch.shape[0]
        batches += 1
    elapsed = time.perf_counter() - t0

    sps = samples / elapsed
    gbs = (samples * BYTES_PER_SAMPLE) / elapsed / 1e9
    ms_batch = elapsed / batches * 1000
    print(f"  {label:<50s}: {sps:>6.0f} samp/s  {gbs:.3f} GB/s  {ms_batch:.0f} ms/batch")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run write + DataLoader benchmark."""
    print()
    print("=" * 72)
    print(
        f"  Zarr DataLoader Benchmark — {N_SAMPLES} × {MATRIX_ROWS}×{MATRIX_COLS} float32  ({TOTAL_GB:.2f} GB)"
    )
    print("=" * 72)
    print()

    rng = np.random.default_rng(0)
    data = rng.random((N_SAMPLES, MATRIX_ROWS, MATRIX_COLS)).astype(DTYPE)

    bench_root = Path("/home/archer/projects/dlkit/output")
    bench_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=bench_root) as tmp:
        tmp_path = Path(tmp)

        # --- write ---
        _header("WRITE (uncompressed)")
        for cs in (1, 32, 128):
            p = tmp_path / f"pack_chunk{cs}"
            t0 = time.perf_counter()
            _write_pack(p, data, chunk_size=cs)
            s = time.perf_counter() - t0
            print(f"  chunk_size={cs:<4d}: {s:.2f} s  {TOTAL_GB * 1e3 / s:.0f} MB/s")

        del data
        print()

        # --- DataLoader ---
        nw = min(4, os.cpu_count() or 1)
        _header(f"DATALOADER  batch={BATCH_SIZE}  workers=0 / {nw}")

        for cs in (1, 32, 128):
            reader = ZarrDensePackReader(tmp_path / f"pack_chunk{cs}")
            print(f"\n  chunk_size={cs}:")
            _measure(reader, num_workers=0, shuffle=True, label="  random shuffle   workers=0")
            _measure(reader, num_workers=nw, shuffle=True, label=f"  random shuffle   workers={nw}")
            _measure(reader, num_workers=0, shuffle=False, label="  sequential      workers=0")
            _measure(reader, num_workers=nw, shuffle=False, label=f"  sequential      workers={nw}")

        print()

    print("Done.")


if __name__ == "__main__":
    main()
