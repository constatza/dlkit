"""Tests for `RunSettings.resolve_seed` and the standalone `apply_run_context`."""

from __future__ import annotations

import torch

from dlkit.infrastructure.config.run_settings import RunSettings, apply_run_context
from dlkit.infrastructure.precision.context import current_precision_override
from dlkit.infrastructure.precision.strategy import PrecisionStrategy


def test_resolve_seed_returns_explicit_seed_when_set() -> None:
    run = RunSettings(seed=7)

    assert run.resolve_seed() == 7


def test_resolve_seed_returns_default_when_unset() -> None:
    run = RunSettings(seed=None)

    assert run.resolve_seed() == 42


def test_resolve_seed_respects_custom_default() -> None:
    run = RunSettings(seed=None)

    assert run.resolve_seed(default=123) == 123


def test_apply_run_context_yields_resolved_seed() -> None:
    run = RunSettings(seed=99)

    with apply_run_context(run) as seed:
        assert seed == 99


def test_apply_run_context_seeds_global_rng_reproducibly() -> None:
    """Entering the context with the same seed must leave the RNG in the same state."""
    run = RunSettings(seed=11)

    with apply_run_context(run):
        first_draw = torch.rand(1)

    with apply_run_context(run):
        second_draw = torch.rand(1)

    assert torch.equal(first_draw, second_draw)


def test_apply_run_context_applies_configured_precision_override() -> None:
    run = RunSettings(seed=1, precision=PrecisionStrategy.FULL_64)

    assert current_precision_override() is None
    with apply_run_context(run):
        assert current_precision_override() == PrecisionStrategy.FULL_64
    assert current_precision_override() is None


def test_apply_run_context_seeds_only_when_precision_unset() -> None:
    run = RunSettings(seed=1, precision=None)

    with apply_run_context(run):
        assert current_precision_override() is None
