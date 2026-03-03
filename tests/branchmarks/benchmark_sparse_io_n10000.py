"""Benchmark: dense vs sparse FlexibleDataset operation timings.

Run:
    UV_CACHE_DIR=/tmp/uv-cache uv run python tests/branchmarks/benchmark_sparse_io_n10000.py
"""

from __future__ import annotations

import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from dlkit.core.datasets.flexible import FlexibleDataset, collate_tensordict
from dlkit.core.training.functional import relative_energy_norm_loss
from dlkit.tools.config.data_entries import Feature, SparseFeature, Target
from dlkit.tools.io.sparse import open_sparse_pack, save_sparse_pack


def log_benchmark(name: str, **payload: object) -> None:
    """Emit one benchmark metric line with info-level logging."""
    logger.info("benchmark={} {}", name, payload)


def make_sparse_pack(path: Path, n_samples: int, d: int, nnz_per_sample: int, seed: int) -> None:
    """Create deterministic sparse COO benchmark payload."""
    rng = np.random.default_rng(seed)
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    ptr = [0]

    for _ in range(n_samples):
        rows = rng.integers(0, d, size=nnz_per_sample, dtype=np.int64)
        cols = rng.integers(0, d, size=nnz_per_sample, dtype=np.int64)
        vals = rng.standard_normal(nnz_per_sample).astype(np.float32)
        row_parts.append(rows)
        col_parts.append(cols)
        val_parts.append(vals)
        ptr.append(ptr[-1] + nnz_per_sample)

    indices = np.vstack([np.concatenate(row_parts), np.concatenate(col_parts)]).astype(np.int64)
    values = np.concatenate(val_parts).astype(np.float32)
    nnz_ptr = np.asarray(ptr, dtype=np.int64)
    save_sparse_pack(path, indices=indices, values=values, nnz_ptr=nnz_ptr, size=(d, d))


def iter_loader(dataset: FlexibleDataset, batch_size: int) -> tuple[int, int, float]:
    """Iterate one full epoch and return (steps, n_seen, elapsed_seconds)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_tensordict)
    t0 = perf_counter()
    n_seen = 0
    steps = 0
    for batch in loader:
        _ = batch["features"]
        _ = batch["targets"]
        n_seen += int(batch.batch_size[0])
        steps += 1
    return steps, n_seen, perf_counter() - t0


def time_once(name: str, fn) -> tuple[object, float]:
    """Time a one-off operation and log elapsed seconds."""
    t0 = perf_counter()
    result = fn()
    elapsed = perf_counter() - t0
    log_benchmark(name, seconds=elapsed)
    return result, elapsed


def time_repeated(name: str, repeats: int, fn) -> tuple[float, float]:
    """Time repeated operation and log seconds + milliseconds per repeat."""
    t0 = perf_counter()
    sink = 0.0
    for i in range(repeats):
        value = fn(i)
        sink += float(value) if isinstance(value, (int, float)) else 1.0
    elapsed = perf_counter() - t0
    ms_per_repeat = elapsed * 1000.0 / repeats
    log_benchmark(
        name,
        repeats=repeats,
        seconds=elapsed,
        ms_per_repeat=ms_per_repeat,
        sink=sink,
    )
    return elapsed, ms_per_repeat


def main() -> None:
    """Run dense/sparse operation-level benchmark (N close to full-batch size)."""
    n_samples = 628
    matrix_dim = 256
    feature_dim = 32
    nnz_per_sample = 512
    batch_size = 628
    repeats = 200

    log_benchmark(
        "config",
        N=n_samples,
        D=matrix_dim,
        F=feature_dim,
        NNZ=nnz_per_sample,
        BATCH=batch_size,
        REPEATS=repeats,
    )

    root = Path("/tmp/dlkit_io_bench_d256_n628_b628")
    root.mkdir(parents=True, exist_ok=True)
    pack_path = root / "pack_full"
    if pack_path.exists():
        shutil.rmtree(pack_path)
    _, _ = time_once(
        "sparse_pack_build_once",
        lambda: make_sparse_pack(
            pack_path,
            n_samples=n_samples,
            d=matrix_dim,
            nnz_per_sample=nnz_per_sample,
            seed=13,
        ),
    )

    x = torch.randn(n_samples, feature_dim)
    y = torch.randn(n_samples, 1)

    dense_dataset, dense_init_elapsed = time_once(
        "dataset_init_dense_once",
        lambda: FlexibleDataset(
            features=[Feature(name="x", value=x)],
            targets=[Target(name="y", value=y)],
        ),
    )

    sparse_dataset, sparse_init_elapsed = time_once(
        "dataset_init_sparse_once",
        lambda: FlexibleDataset(
            features=[
                Feature(name="x", value=x),
                SparseFeature(name="matrix", path=pack_path, model_input=False, loss_input="matrix"),
            ],
            targets=[Target(name="y", value=y)],
        ),
    )

    reader, _ = time_once("open_sparse_reader_once", lambda: open_sparse_pack(pack_path))

    full_batch_indices = list(range(batch_size))

    _, dense_getitems_ms = time_repeated(
        "dense_getitems_repeated",
        repeats,
        lambda _: int(dense_dataset.__getitems__(full_batch_indices).batch_size[0]),
    )
    _, sparse_getitems_ms = time_repeated(
        "sparse_getitems_repeated",
        repeats,
        lambda _: int(sparse_dataset.__getitems__(full_batch_indices).batch_size[0]),
    )

    sparse_prebatched = sparse_dataset.__getitems__(full_batch_indices)
    dense_prebatched = dense_dataset.__getitems__(full_batch_indices)
    dense_collate_elapsed, dense_collate_ms = time_repeated(
        "dense_collate_prebatched_repeated",
        repeats,
        lambda _: int(collate_tensordict(dense_prebatched).batch_size[0]),
    )
    sparse_collate_elapsed, sparse_collate_ms = time_repeated(
        "sparse_collate_prebatched_repeated",
        repeats,
        lambda _: int(collate_tensordict(sparse_prebatched).batch_size[0]),
    )

    reader_stacked_elapsed, reader_stacked_ms = time_repeated(
        "reader_build_sparse_stacked_repeated",
        repeats,
        lambda _: int(reader.build_torch_sparse_stacked(full_batch_indices, coalesce=False).shape[0]),
    )

    (dense_loader_stats, dense_loader_elapsed) = time_once(
        "loader_iter_dense_once",
        lambda: iter_loader(dense_dataset, batch_size),
    )
    (sparse_loader_stats, sparse_loader_elapsed) = time_once(
        "loader_iter_sparse_once",
        lambda: iter_loader(sparse_dataset, batch_size),
    )
    dense_steps, dense_seen, _ = dense_loader_stats
    sparse_steps, sparse_seen, _ = sparse_loader_stats
    log_benchmark(
        "loader_iter_dense_once_stats",
        steps=dense_steps,
        n_seen=dense_seen,
        sec_per_step=dense_loader_elapsed / max(dense_steps, 1),
    )
    log_benchmark(
        "loader_iter_sparse_once_stats",
        steps=sparse_steps,
        n_seen=sparse_seen,
        sec_per_step=sparse_loader_elapsed / max(sparse_steps, 1),
    )

    matrix_sparse = sparse_prebatched["features"]["matrix"]
    matrix_dense, dense_matrix_materialize_elapsed = time_once(
        "sparse_matrix_to_dense_once",
        lambda: matrix_sparse.to_dense(),
    )
    preds = torch.randn(batch_size, matrix_dim)
    target = torch.randn(batch_size, matrix_dim)

    rel_energy_sparse_elapsed, rel_energy_sparse_ms = time_repeated(
        "relative_energy_norm_sparse_matrix_repeated",
        repeats,
        lambda _: float(relative_energy_norm_loss(preds, target, matrix_sparse).item()),
    )
    rel_energy_dense_elapsed, rel_energy_dense_ms = time_repeated(
        "relative_energy_norm_dense_matrix_repeated",
        repeats,
        lambda _: float(relative_energy_norm_loss(preds, target, matrix_dense).item()),
    )

    # Absolute-value report: which operation actually dominates the budget.
    estimated_sparse_step_ms = sparse_getitems_ms + rel_energy_sparse_ms
    estimated_dense_step_ms = dense_getitems_ms + rel_energy_dense_ms
    log_benchmark(
        "absolute_report",
        dense_getitems_ms=dense_getitems_ms,
        sparse_getitems_ms=sparse_getitems_ms,
        dense_collate_ms=dense_collate_ms,
        sparse_collate_ms=sparse_collate_ms,
        reader_stacked_ms=reader_stacked_ms,
        relative_energy_sparse_ms=rel_energy_sparse_ms,
        relative_energy_dense_ms=rel_energy_dense_ms,
        relative_energy_over_sparse_getitems=rel_energy_sparse_ms / max(sparse_getitems_ms, 1e-12),
        relative_energy_over_dense_getitems=rel_energy_dense_ms / max(dense_getitems_ms, 1e-12),
        estimated_sparse_step_ms=estimated_sparse_step_ms,
        estimated_dense_step_ms=estimated_dense_step_ms,
        sparse_getitems_share_of_estimated_step=sparse_getitems_ms / max(estimated_sparse_step_ms, 1e-12),
        relative_energy_share_of_estimated_sparse_step=rel_energy_sparse_ms
        / max(estimated_sparse_step_ms, 1e-12),
    )

    log_benchmark(
        "comparison_ratios",
        sparse_over_dense_getitems_ratio=sparse_getitems_ms / max(dense_getitems_ms, 1e-12),
        sparse_over_dense_loader_ratio=sparse_loader_elapsed / max(dense_loader_elapsed, 1e-12),
        sparse_over_dense_rel_energy_ratio=rel_energy_sparse_ms / max(rel_energy_dense_ms, 1e-12),
        sparse_init_over_dense_init_ratio=sparse_init_elapsed / max(dense_init_elapsed, 1e-12),
        dense_matrix_materialize_once_s=dense_matrix_materialize_elapsed,
        relative_energy_sparse_elapsed_s=rel_energy_sparse_elapsed,
        relative_energy_dense_elapsed_s=rel_energy_dense_elapsed,
        reader_stacked_elapsed_s=reader_stacked_elapsed,
        dense_collate_elapsed_s=dense_collate_elapsed,
        sparse_collate_elapsed_s=sparse_collate_elapsed,
    )

    max_sparse_over_dense_ratio = 5.0
    repeated_ratio = sparse_getitems_ms / max(dense_getitems_ms, 1e-12)
    loader_ratio = sparse_loader_elapsed / max(dense_loader_elapsed, 1e-12)
    log_benchmark(
        "assertion",
        metric="sparse_over_dense_loader_ratio",
        max_allowed=max_sparse_over_dense_ratio,
        observed=loader_ratio,
        diagnostic_sparse_over_dense_getitems_ratio=repeated_ratio,
    )
    if loader_ratio > max_sparse_over_dense_ratio:
        raise AssertionError(
            f"Sparse loader regression: ratio={loader_ratio:.3f} exceeds "
            f"limit={max_sparse_over_dense_ratio:.3f}"
        )


if __name__ == "__main__":
    main()
