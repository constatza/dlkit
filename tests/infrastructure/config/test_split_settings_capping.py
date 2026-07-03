"""Tests for IndexSplitSettings capping fields.

Covers:
- Default values: max_train_samples=None, train_subset_seed=None
- Round-trip through Pydantic with max_train_samples set
- Property accessors (train_ratio, has_existing_split)
"""

from __future__ import annotations

import pytest

from dlkit.infrastructure.config.split_settings import IndexSplitSettings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TEST_RATIO: float = 0.15
DEFAULT_VAL_RATIO: float = 0.15
CAP_SAMPLES: int = 50
FIXED_SEED: int = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_split_settings() -> IndexSplitSettings:
    """IndexSplitSettings with all defaults — no filepath, no cap.

    Returns:
        IndexSplitSettings: Minimal instance with only ratio defaults.
    """
    return IndexSplitSettings()


@pytest.fixture
def capped_split_settings() -> IndexSplitSettings:
    """IndexSplitSettings with max_train_samples cap set.

    Returns:
        IndexSplitSettings: Instance with max_train_samples=50.
    """
    return IndexSplitSettings(max_train_samples=CAP_SAMPLES)


@pytest.fixture
def capped_and_seeded_split_settings() -> IndexSplitSettings:
    """IndexSplitSettings with both cap and seed set.

    Returns:
        IndexSplitSettings: Instance with max_train_samples=50, train_subset_seed=42.
    """
    return IndexSplitSettings(max_train_samples=CAP_SAMPLES, train_subset_seed=FIXED_SEED)


# ---------------------------------------------------------------------------
# Good-path tests
# ---------------------------------------------------------------------------


def test_default_max_train_samples_is_none(default_split_settings: IndexSplitSettings) -> None:
    """max_train_samples defaults to None.

    Args:
        default_split_settings: IndexSplitSettings with all defaults.
    """
    assert default_split_settings.max_train_samples is None


def test_default_train_subset_seed_is_none(default_split_settings: IndexSplitSettings) -> None:
    """train_subset_seed defaults to None.

    Args:
        default_split_settings: IndexSplitSettings with all defaults.
    """
    assert default_split_settings.train_subset_seed is None


def test_default_has_no_existing_split(default_split_settings: IndexSplitSettings) -> None:
    """has_existing_split is False when filepath is None.

    Args:
        default_split_settings: IndexSplitSettings with all defaults.
    """
    assert default_split_settings.has_existing_split is False


def test_default_train_ratio(default_split_settings: IndexSplitSettings) -> None:
    """train_ratio computes correctly from defaults.

    Args:
        default_split_settings: IndexSplitSettings with default ratios 0.15/0.15.
    """
    expected = 1.0 - DEFAULT_TEST_RATIO - DEFAULT_VAL_RATIO
    assert abs(default_split_settings.train_ratio - expected) < 1e-9


def test_capped_settings_stores_max_train_samples(
    capped_split_settings: IndexSplitSettings,
) -> None:
    """max_train_samples is stored correctly after Pydantic round-trip.

    Args:
        capped_split_settings: IndexSplitSettings with max_train_samples=50.
    """
    assert capped_split_settings.max_train_samples == CAP_SAMPLES


def test_capped_settings_seed_still_none(capped_split_settings: IndexSplitSettings) -> None:
    """train_subset_seed remains None when only max_train_samples is provided.

    Args:
        capped_split_settings: IndexSplitSettings with max_train_samples=50.
    """
    assert capped_split_settings.train_subset_seed is None


def test_seeded_settings_stores_both_fields(
    capped_and_seeded_split_settings: IndexSplitSettings,
) -> None:
    """Both max_train_samples and train_subset_seed are stored correctly.

    Args:
        capped_and_seeded_split_settings: Instance with cap and seed.
    """
    assert capped_and_seeded_split_settings.max_train_samples == CAP_SAMPLES
    assert capped_and_seeded_split_settings.train_subset_seed == FIXED_SEED


def test_get_split_ratios_sums_near_one(default_split_settings: IndexSplitSettings) -> None:
    """get_split_ratios() returns a three-tuple that sums to approximately 1.0.

    Args:
        default_split_settings: IndexSplitSettings with default ratios.
    """
    train, val, test = default_split_settings.get_split_ratios()
    assert abs(train + val + test - 1.0) < 1e-9
