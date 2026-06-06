"""Tests for split resolution without implicit local persistence."""

from pathlib import Path

import pytest

from dlkit.infrastructure.io.split_provider import SplitResolution, get_or_create_split
from dlkit.infrastructure.types.split import IndexSplit


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
    """Generated splits stay in memory and expose deterministic artifact names."""
    resolution = get_or_create_split(
        num_samples=100,
        test_ratio=0.2,
        val_ratio=0.2,
        session_name="test_session",
    )

    assert isinstance(resolution, SplitResolution)
    split = resolution.index_split
    assert isinstance(split, IndexSplit)
    assert len(split.train) + len(split.validation) + len(split.test) == 100
    assert resolution.source_path is None
    assert resolution.artifact_filename == "test_session_100_split.json"
    assert not (tmp_path / resolution.artifact_filename).exists()


def test_get_or_create_split_uses_explicit_filepath(
    explicit_split_file: Path, sample_split_data: dict
):
    """Test that explicit filepath takes precedence."""
    resolution = get_or_create_split(
        num_samples=100,
        test_ratio=0.2,
        val_ratio=0.2,
        session_name="ignored",
        explicit_filepath=explicit_split_file,
    )

    split = resolution.index_split
    # Should match the explicit file data
    assert list(split.train) == sample_split_data["train"]
    assert list(split.validation) == sample_split_data["validation"]
    assert list(split.test) == sample_split_data["test"]
    assert split.predict is not None
    assert list(split.predict) == sample_split_data["predict"]
    assert resolution.source_path == explicit_split_file
    assert resolution.artifact_filename == explicit_split_file.name


def test_get_or_create_split_different_sessions_create_different_splits():
    """Session name affects artifact naming, not split generation semantics."""
    split1 = get_or_create_split(
        num_samples=50,
        test_ratio=0.2,
        val_ratio=0.2,
        session_name="session_a",
    ).index_split

    split2 = get_or_create_split(
        num_samples=50,
        test_ratio=0.2,
        val_ratio=0.2,
        session_name="session_b",
    ).index_split

    # Different sessions should have different random splits
    # (very unlikely to be identical)
    assert split1.train != split2.train


def test_get_or_create_split_rejects_corrupt_explicit_file(tmp_path: Path):
    """Corrupt explicit split files fail fast instead of silently regenerating."""
    split_file = tmp_path / "corrupt_split.json"
    split_file.write_text("not valid json{{{")

    with pytest.raises(Exception):
        get_or_create_split(
            num_samples=50,
            test_ratio=0.2,
            val_ratio=0.2,
            session_name="corrupt_test",
            explicit_filepath=split_file,
        )


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
    ).index_split

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
    ).index_split

    # Combine all indices
    all_indices = set(split.train) | set(split.validation) | set(split.test)
    if split.predict:
        all_indices |= set(split.predict)

    # Should cover all samples exactly once
    assert len(all_indices) == num_samples
    assert all_indices == set(range(num_samples))
