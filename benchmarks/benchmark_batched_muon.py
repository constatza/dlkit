"""Benchmark: torch.optim.Muon vs BatchedMuon vs Adam (optimizer step only).

Measures wall-clock time for optimizer.step() in isolation (gradients
pre-computed) to isolate optimizer overhead from forward/backward cost.

Usage::

    uv run python benchmarks/benchmark_batched_muon.py

Sample results (CPU, 6 × 256×256 weight matrices):

    Adam (all params)         :    0.727 ms/step
    torch.optim.Muon + AdamW  : 1036.280 ms/step   (1426x vs Adam)
    BatchedMuon + AdamW       : 1054.200 ms/step   (1451x vs Adam)
    Speedup Muon → Batched    :    1.0x

No speedup on CPU (BLAS saturates cores per matmul regardless of batching).
On GPU the speedup is meaningful: batching eliminates per-parameter
kernel-launch overhead (~1–10 µs each on CUDA).  The benchmark runs on
CUDA automatically when ``torch.cuda.is_available()`` is True.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from dlkit.engine.training.optimization.batched_muon import BatchedMuon

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 10
ITERS = 100


def bench(name: str, fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Time ``fn`` over ``iters`` calls after ``warmup`` warm-up calls.

    Args:
        name: Label printed in the results table.
        fn: Zero-argument callable to benchmark.
        warmup: Number of un-timed warm-up iterations.
        iters: Number of timed iterations.

    Returns:
        Mean wall-clock time per call in milliseconds.
    """
    for _ in range(warmup):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {name:<40s}: {ms:.3f} ms/step")
    return ms


def make_model(width: int = 256, depth: int = 6) -> nn.Sequential:
    """Build a plain FFN with ``depth`` hidden layers of ``width`` units.

    Args:
        width: Hidden layer width (all layers square).
        depth: Number of Linear+ReLU pairs.

    Returns:
        Sequential model on ``DEVICE``.
    """
    return nn.Sequential(
        *[layer for _ in range(depth) for layer in (nn.Linear(width, width, bias=True), nn.ReLU())]
    ).to(DEVICE)


def main() -> None:
    model = make_model()
    x = torch.randn(64, 256, device=DEVICE)

    muon_params = [p for p in model.parameters() if p.ndim == 2]
    other_params = [p for p in model.parameters() if p.ndim != 2]

    print(
        f"\nDevice: {DEVICE}  |  "
        f"Muon-eligible params: {len(muon_params)}  |  "
        f"Other: {len(other_params)}\n"
    )

    # Pre-compute gradients once; reused across all optimizer timing calls.
    model(x).sum().backward()

    print("=== optimizer.step() only (gradients pre-computed) ===")

    adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    adam_ms = bench("Adam (all params)", adam.step)

    ref_muon = torch.optim.Muon(muon_params, lr=1e-3)
    ref_adamw = torch.optim.AdamW(other_params, lr=1e-3)

    def ref_step() -> None:
        ref_muon.step()
        ref_adamw.step()

    ref_ms = bench("torch.optim.Muon + AdamW", ref_step)

    bat_muon = BatchedMuon(muon_params, lr=1e-3)
    bat_adamw = torch.optim.AdamW(other_params, lr=1e-3)

    def bat_step() -> None:
        bat_muon.step()
        bat_adamw.step()

    bat_ms = bench("BatchedMuon + AdamW", bat_step)

    print()
    print(f"  torch.optim.Muon / Adam   : {ref_ms / adam_ms:.1f}x slower")
    print(f"  BatchedMuon      / Adam   : {bat_ms / adam_ms:.1f}x slower")
    print(f"  Speedup (Muon → Batched)  : {ref_ms / bat_ms:.1f}x")
    print()


if __name__ == "__main__":
    main()
