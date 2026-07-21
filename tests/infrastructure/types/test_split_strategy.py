"""Structural tests for RatioSplitStrategy.

RatioSplitStrategy is fully determined by its seed (same seed twice ->
identical split; different seeds -> different split), decoupled from
ambient global RNG state.

ExternalFileSplitStrategy is tested in
``tests/infrastructure/io/test_split_provider.py`` instead — it lives in
``infrastructure.io.split_provider``, not here, to avoid a ``types -> io``
dependency cycle (see that strategy's docstring).
"""

from __future__ import annotations

import torch

from dlkit.infrastructure.types.split import RatioSplitStrategy

NUM_SAMPLES: int = 200
TEST_RATIO: float = 0.15
VAL_RATIO: float = 0.15
SEED_A: int = 111
SEED_B: int = 222


def test_ratio_split_strategy_same_seed_is_deterministic() -> None:
    """Two RatioSplitStrategy instances with the same seed produce identical splits."""
    first = RatioSplitStrategy(
        num_samples=NUM_SAMPLES, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, seed=SEED_A
    ).split()
    second = RatioSplitStrategy(
        num_samples=NUM_SAMPLES, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, seed=SEED_A
    ).split()

    assert first == second


def test_ratio_split_strategy_different_seeds_differ() -> None:
    """RatioSplitStrategy instances with different seeds produce different splits."""
    first = RatioSplitStrategy(
        num_samples=NUM_SAMPLES, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, seed=SEED_A
    ).split()
    second = RatioSplitStrategy(
        num_samples=NUM_SAMPLES, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, seed=SEED_B
    ).split()

    assert first != second


def test_ratio_split_strategy_ignores_ambient_global_rng_state() -> None:
    """The seeded split does not depend on ambient torch global RNG state.

    Directly targets the original bug: RatioSplitStrategy used to call
    torch.randperm() against global state with no seed of its own, so
    repeated calls (or calls preceded by unrelated RNG draws elsewhere in
    the pipeline) produced different splits. A local torch.Generator fixes
    this — the split must be identical regardless of prior global RNG state.
    """
    torch.manual_seed(0)
    first = RatioSplitStrategy(
        num_samples=NUM_SAMPLES, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, seed=SEED_A
    ).split()

    torch.manual_seed(999)
    torch.randperm(50)  # Consume some ambient global RNG state.
    second = RatioSplitStrategy(
        num_samples=NUM_SAMPLES, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, seed=SEED_A
    ).split()

    assert first == second
