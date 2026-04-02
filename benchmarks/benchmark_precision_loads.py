"""Micro-benchmark for repeated precision-aware loads.

Run:
    uv run python benchmarks/benchmark_precision_loads.py
"""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import torch

from dlkit.tools.config.precision import PrecisionStrategy
from dlkit.tools.config.precision.context import precision_override
from dlkit.tools.io.arrays import load_array


def _bench_root() -> Path:
    return Path(os.environ.get("DLKIT_BENCH_ROOT", "/tmp/dlkit-benchmarks")) / "precision"


def _prepare_tensor_file() -> Path:
    root = _bench_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "tensor.pt"
    tensor = torch.randn(512, 128, dtype=torch.float64)
    torch.save(tensor, path)
    return path


def _time_loads(path: Path, strategy: PrecisionStrategy, repeats: int) -> float:
    t0 = perf_counter()
    with precision_override(strategy):
        for _ in range(repeats):
            tensor = load_array(path)
            _ = tensor.dtype
    return perf_counter() - t0


def main() -> None:
    path = _prepare_tensor_file()
    repeats = 50
    for strategy in (
        PrecisionStrategy.FULL_32,
        PrecisionStrategy.FULL_64,
        PrecisionStrategy.MIXED_16,
    ):
        elapsed = _time_loads(path, strategy, repeats)
        print(
            f"strategy={strategy.value} repeats={repeats} "
            f"seconds={elapsed:.4f} ms_per_load={(elapsed * 1000.0 / repeats):.2f}"
        )


if __name__ == "__main__":
    main()
