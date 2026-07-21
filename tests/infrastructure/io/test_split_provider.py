"""Structural tests for the producer/consumer split-resolution pair.

Supersedes the deleted, temporary `test_split_determinism.py` (which proved
the pre-fix `get_or_create_split()` was unseeded). Covers:
- resolve_training_split() is deterministic given a fixed seed (the fix
  itself, exercised through the producer function rather than the strategy
  directly).
- resolve_evaluation_split() has no ratio/seed parameters at all (the
  Interface Segregation guarantee that a caller cannot accidentally fall
  back to regenerating a split), enforced via `inspect.signature`.
- resolve_evaluation_split() propagates a loud failure for a missing file.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from dlkit.infrastructure.io.index import save_split_indices
from dlkit.infrastructure.io.split_provider import (
    ExternalFileSplitStrategy,
    resolve_evaluation_split,
    resolve_training_split,
)
from dlkit.infrastructure.types.split import IndexSplit

NUM_SAMPLES: int = 200
TEST_RATIO: float = 0.15
VAL_RATIO: float = 0.15
FIXED_SEED: int = 7


@pytest.fixture
def persisted_training_split(tmp_path: Path) -> Path:
    """A split file produced by resolve_training_split with persist_to set.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to the persisted split file.
    """
    persist_to = tmp_path / "splits" / "session_split.json"
    resolve_training_split(
        num_samples=NUM_SAMPLES,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=FIXED_SEED,
        persist_to=persist_to,
        session_name="session",
    )
    return persist_to


def test_resolve_training_split_is_deterministic_given_a_fixed_seed() -> None:
    """Two resolve_training_split() calls with identical arguments (including
    seed) produce identical splits — the fix for the original unseeded bug.
    """
    first = resolve_training_split(
        num_samples=NUM_SAMPLES,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=FIXED_SEED,
        persist_to=None,
        session_name="determinism_test",
    )
    second = resolve_training_split(
        num_samples=NUM_SAMPLES,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=FIXED_SEED,
        persist_to=None,
        session_name="determinism_test",
    )

    assert first.index_split == second.index_split


def test_resolve_evaluation_split_reloads_the_persisted_training_split(
    persisted_training_split: Path,
) -> None:
    """resolve_evaluation_split() reloads exactly what resolve_training_split() wrote.

    Args:
        persisted_training_split: Path written by the resolve_training_split fixture.
    """
    resolution = resolve_evaluation_split(filepath=persisted_training_split)

    assert isinstance(resolution.index_split, IndexSplit)
    assert resolution.source_path == persisted_training_split
    assert resolution.artifact_filename == persisted_training_split.name


def test_resolve_evaluation_split_has_no_ratio_or_seed_parameters() -> None:
    """Structural ISP guarantee: resolve_evaluation_split cannot generate a fresh split.

    A consumer call site cannot accidentally fall back to regenerating an
    unrelated random split, because the parameters needed to do so
    (num_samples/test_ratio/val_ratio/seed) do not exist on this function's
    signature at all — not merely defaulted or ignored.
    """
    parameter_names = set(inspect.signature(resolve_evaluation_split).parameters)

    assert parameter_names == {"filepath"}
    for forbidden in ("num_samples", "test_ratio", "val_ratio", "seed"):
        assert forbidden not in parameter_names


def test_resolve_evaluation_split_on_missing_file_raises_loudly(tmp_path: Path) -> None:
    """A missing split file fails loudly instead of silently regenerating one."""
    missing_path = tmp_path / "does_not_exist.json"

    with pytest.raises(Exception):
        resolve_evaluation_split(filepath=missing_path)


def test_resolve_evaluation_split_on_malformed_file_raises_loudly(tmp_path: Path) -> None:
    """A malformed split file fails loudly instead of silently regenerating one."""
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("not valid json{{{")

    with pytest.raises(Exception):
        resolve_evaluation_split(filepath=malformed_path)


def test_external_file_split_strategy_loads_persisted_split(tmp_path: Path) -> None:
    """ExternalFileSplitStrategy reloads a persisted split verbatim.

    Lives in this ``io``-layer test module (not alongside RatioSplitStrategy's
    tests in ``tests/infrastructure/types/``) because ExternalFileSplitStrategy
    itself lives in ``infrastructure.io.split_provider`` — it depends on
    ``load_split_indices``, and colocating it with ``RatioSplitStrategy`` in
    ``infrastructure.types.split`` would create a ``types -> io`` cycle
    (``io`` already depends on ``types`` for ``IndexSplit``/``SplitStrategy``).
    """
    sample_index_split = IndexSplit(
        train=(0, 1, 2, 3, 4, 5),
        validation=(6, 7),
        test=(8, 9),
        predict=None,
    )
    path = tmp_path / "split.json"
    save_split_indices(sample_index_split, path)

    loaded = ExternalFileSplitStrategy(path).split()

    assert loaded == sample_index_split
