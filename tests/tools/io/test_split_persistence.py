"""Tests for split index persistence and caching."""

from pathlib import Path

import pytest

from dlkit.core.datatypes.split import IndexSplit
from dlkit.tools.io.locations import splits_dir
from dlkit.tools.io.split_provider import get_or_create_split


@pytest.fixture
def sample_split_data() -> dict:
    """Sample split data for testing."""
    return {
        "train": [0, 1, 2, 3, 4, 5, 6],
        "validation": [7, 8],
        "test": [9, 10],
        "predict": [11, 12],
    }


@pytest.fixture
def explicit_split_file(tmp_path: Path, sample_split_data: dict) -> Path:
    """Create an explicit split file for testing."""
    import json

    split_file = tmp_path / "custom_split.json"
    with split_file.open("w") as f:
        json.dump(sample_split_data, f)
    return split_file


def test_get_or_create_split_generates_new_split(tmp_path: Path):
    """Test that get_or_create_split generates a new split when none exists."""
    split = get_or_create_split(
        num_samples=100,
        test_ratio=0.2,
        val_ratio=0.2,
        session_name="test_session",
    )

    assert isinstance(split, IndexSplit)
    assert len(split.train) + len(split.validation) + len(split.test) == 100
    # Should have saved the split
    split_file = splits_dir() / "test_session_100_split.json"
    assert split_file.exists()


def test_get_or_create_split_loads_cached_split(tmp_path: Path):
    """Test that get_or_create_split loads an existing cached split."""
    # First call: create and cache
    split1 = get_or_create_split(
        num_samples=100,
        test_ratio=0.2,
        val_ratio=0.2,
        session_name="cache_test",
    )

    # Second call: should load from cache (same indices)
    split2 = get_or_create_split(
        num_samples=100,
        test_ratio=0.2,
        val_ratio=0.2,
        session_name="cache_test",
    )

    # Verify same split was loaded
    assert split1.train == split2.train
    assert split1.validation == split2.validation
    assert split1.test == split2.test


def test_get_or_create_split_uses_explicit_filepath(
    explicit_split_file: Path, sample_split_data: dict
):
    """Test that explicit filepath takes precedence."""
    split = get_or_create_split(
        num_samples=100,
        test_ratio=0.2,
        val_ratio=0.2,
        session_name="ignored",
        explicit_filepath=explicit_split_file,
    )

    # Should match the explicit file data
    assert list(split.train) == sample_split_data["train"]
    assert list(split.validation) == sample_split_data["validation"]
    assert list(split.test) == sample_split_data["test"]
    assert split.predict is not None
    assert list(split.predict) == sample_split_data["predict"]


def test_get_or_create_split_different_sessions_create_different_splits():
    """Test that different session names create different cached splits."""
    split1 = get_or_create_split(
        num_samples=50,
        test_ratio=0.2,
        val_ratio=0.2,
        session_name="session_a",
    )

    split2 = get_or_create_split(
        num_samples=50,
        test_ratio=0.2,
        val_ratio=0.2,
        session_name="session_b",
    )

    # Different sessions should have different random splits
    # (very unlikely to be identical)
    assert split1.train != split2.train


def test_get_or_create_split_handles_corrupt_cache(tmp_path: Path):
    """Test that corrupt cache files are regenerated."""
    # Create a corrupt cache file
    split_file = splits_dir() / "corrupt_test_50_split.json"
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with split_file.open("w") as f:
        f.write("not valid json{{{")

    # Should handle corruption and generate new split
    split = get_or_create_split(
        num_samples=50,
        test_ratio=0.2,
        val_ratio=0.2,
        session_name="corrupt_test",
    )

    assert isinstance(split, IndexSplit)
    assert len(split.train) + len(split.validation) + len(split.test) == 50


def test_split_ratios_are_respected():
    """Test that split ratios are approximately correct."""
    num_samples = 1000
    test_ratio = 0.15
    val_ratio = 0.15

    split = get_or_create_split(
        num_samples=num_samples,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        session_name="ratio_test",
    )

    # Check ratios are approximately correct (within 1%)
    actual_test_ratio = len(split.test) / num_samples
    actual_val_ratio = len(split.validation) / num_samples
    actual_train_ratio = len(split.train) / num_samples

    assert abs(actual_test_ratio - test_ratio) < 0.01
    assert abs(actual_val_ratio - val_ratio) < 0.01
    assert abs(actual_train_ratio - (1 - test_ratio - val_ratio)) < 0.01


def test_split_indices_are_unique_and_complete():
    """Test that split indices don't overlap and cover all samples."""
    num_samples = 100
    split = get_or_create_split(
        num_samples=num_samples,
        test_ratio=0.2,
        val_ratio=0.2,
        session_name="unique_test",
    )

    # Combine all indices
    all_indices = set(split.train) | set(split.validation) | set(split.test)
    if split.predict:
        all_indices |= set(split.predict)

    # Should cover all samples exactly once
    assert len(all_indices) == num_samples
    assert all_indices == set(range(num_samples))
