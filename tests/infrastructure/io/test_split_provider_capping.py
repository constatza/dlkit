"""Tests for resolve_training_split() capping logic.

Covers:
- Without cap: train size matches expected proportion
- With max_train_samples=10: train is truncated to 10
- With max_train_samples larger than pool: no truncation, no error
- Nesting invariant: capped train is a prefix of full train (seed=None)
- Val/test indices are identical regardless of max_train_samples
"""

from __future__ import annotations

import pytest

from dlkit.infrastructure.io.split_provider import SplitResolution, resolve_training_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_SAMPLES: int = 200
TEST_RATIO: float = 0.15
VAL_RATIO: float = 0.15
SESSION_NAME: str = "test_session"
FIXED_SEED: int = 42

# Expected train count without any cap (floor-truncating ratios)
_TEST_COUNT: int = int(NUM_SAMPLES * TEST_RATIO)
_VAL_COUNT: int = int(NUM_SAMPLES * VAL_RATIO)
EXPECTED_TRAIN_COUNT: int = NUM_SAMPLES - _TEST_COUNT - _VAL_COUNT

SMALL_CAP: int = 10
LARGE_CAP: int = 10_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _resolve(*, max_train_samples: int | None = None, train_subset_seed: int | None = None):
    """Resolve a training split with the module's fixed seed, uncapped by default.

    Args:
        max_train_samples: Optional cap on the number of training samples.
        train_subset_seed: Optional seed for re-permuting train indices before capping.

    Returns:
        SplitResolution for NUM_SAMPLES samples with FIXED_SEED.
    """
    return resolve_training_split(
        num_samples=NUM_SAMPLES,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=FIXED_SEED,
        persist_to=None,
        session_name=SESSION_NAME,
        max_train_samples=max_train_samples,
        train_subset_seed=train_subset_seed,
    )


@pytest.fixture
def uncapped_split() -> SplitResolution:
    """SplitResolution without any cap on training samples.

    Returns:
        SplitResolution: Full split for NUM_SAMPLES samples.
    """
    return _resolve()


@pytest.fixture
def small_capped_split() -> SplitResolution:
    """SplitResolution with max_train_samples=10.

    Returns:
        SplitResolution: Split where train is capped at 10 samples.
    """
    return _resolve(max_train_samples=SMALL_CAP)


@pytest.fixture
def large_capped_split() -> SplitResolution:
    """SplitResolution with max_train_samples larger than available train pool.

    Returns:
        SplitResolution: Split where cap exceeds train pool, so no truncation occurs.
    """
    return _resolve(max_train_samples=LARGE_CAP)


# ---------------------------------------------------------------------------
# Good-path tests
# ---------------------------------------------------------------------------


def test_uncapped_split_train_size_matches_proportion(
    uncapped_split: SplitResolution,
) -> None:
    """Without a cap, the train split has the expected proportional size.

    Args:
        uncapped_split: Full split fixture.
    """
    assert len(uncapped_split.index_split.train) == EXPECTED_TRAIN_COUNT


def test_small_cap_truncates_train_to_cap(small_capped_split: SplitResolution) -> None:
    """With max_train_samples=10, train split is exactly 10.

    Args:
        small_capped_split: Split fixture capped at SMALL_CAP.
    """
    assert len(small_capped_split.index_split.train) == SMALL_CAP


def test_large_cap_does_not_error_and_returns_full_train(
    large_capped_split: SplitResolution,
) -> None:
    """A cap larger than the train pool returns the full train split without error.

    Args:
        large_capped_split: Split fixture with cap exceeding available samples.
    """
    assert len(large_capped_split.index_split.train) == EXPECTED_TRAIN_COUNT


def test_nesting_invariant_capped_is_prefix_of_full() -> None:
    """With train_subset_seed=None, capped train is a prefix of the full train split.

    The same seed produces the same permutation, so the first SMALL_CAP
    indices of the uncapped split must equal the capped split's train indices.
    """
    full = _resolve()
    capped = _resolve(max_train_samples=SMALL_CAP, train_subset_seed=None)

    full_prefix = full.index_split.train[:SMALL_CAP]
    assert capped.index_split.train == full_prefix


def test_val_indices_identical_regardless_of_cap() -> None:
    """Validation indices are identical whether or not a cap is applied."""
    full = _resolve()
    capped = _resolve(max_train_samples=SMALL_CAP)

    assert full.index_split.validation == capped.index_split.validation


def test_test_indices_identical_regardless_of_cap() -> None:
    """Test indices are identical whether or not a cap is applied."""
    full = _resolve()
    capped = _resolve(max_train_samples=SMALL_CAP)

    assert full.index_split.test == capped.index_split.test


def test_split_resolution_has_no_source_path_when_generated(
    uncapped_split: SplitResolution,
) -> None:
    """In-memory generated splits have source_path=None.

    Args:
        uncapped_split: Full split fixture (no explicit filepath).
    """
    assert uncapped_split.source_path is None
    assert not uncapped_split.has_explicit_file


def test_artifact_filename_contains_session_name(uncapped_split: SplitResolution) -> None:
    """Artifact filename is derived from the session name.

    Args:
        uncapped_split: Full split fixture.
    """
    assert SESSION_NAME in uncapped_split.artifact_filename
