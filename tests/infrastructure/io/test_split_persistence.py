"""Tests for `resolve_training_split()` — the seeded split producer."""

from pathlib import Path

import pytest

from dlkit.infrastructure.io.split_provider import SplitResolution, resolve_training_split
from dlkit.infrastructure.types.split import IndexSplit

FIXED_SEED: int = 1234
OTHER_SEED: int = 5678


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


@pytest.fixture
def explicit_split_file_toml(tmp_path: Path, sample_split_data: dict) -> Path:
    """Create an explicit TOML split file for testing."""
    split_file = tmp_path / "custom_split.toml"
    lines = [f"{key} = {value!r}" for key, value in sample_split_data.items()]
    split_file.write_text("\n".join(lines))
    return split_file


def test_resolve_training_split_generates_new_split():
    """Generated splits stay in memory (persist_to=None) and expose deterministic naming."""
    resolution = resolve_training_split(
        num_samples=100,
        test_ratio=0.2,
        val_ratio=0.2,
        seed=FIXED_SEED,
        persist_to=None,
        session_name="test_session",
    )

    assert isinstance(resolution, SplitResolution)
    split = resolution.index_split
    assert isinstance(split, IndexSplit)
    assert len(split.train) + len(split.validation) + len(split.test) == 100
    assert resolution.source_path is None
    assert resolution.artifact_filename == "test_session_100_split.json"


def test_resolve_training_split_persists_when_persist_to_given(tmp_path: Path):
    """When persist_to is given, the resolved split is written to that path."""
    persist_to = tmp_path / "splits" / "session_100_split.json"
    resolution = resolve_training_split(
        num_samples=100,
        test_ratio=0.2,
        val_ratio=0.2,
        seed=FIXED_SEED,
        persist_to=persist_to,
        session_name="session",
    )

    assert persist_to.exists()
    from dlkit.infrastructure.io.index import load_split_indices

    assert load_split_indices(persist_to) == resolution.index_split


def test_resolve_training_split_uses_explicit_filepath(
    explicit_split_file: Path, sample_split_data: dict
):
    """Test that explicit filepath takes precedence over seeded generation."""
    resolution = resolve_training_split(
        num_samples=100,
        test_ratio=0.2,
        val_ratio=0.2,
        seed=FIXED_SEED,
        persist_to=None,
        session_name="ignored",
        explicit_filepath=explicit_split_file,
    )

    split = resolution.index_split
    assert list(split.train) == sample_split_data["train"]
    assert list(split.validation) == sample_split_data["validation"]
    assert list(split.test) == sample_split_data["test"]
    assert split.predict is not None
    assert list(split.predict) == sample_split_data["predict"]
    assert resolution.source_path == explicit_split_file
    assert resolution.artifact_filename == explicit_split_file.name


def test_resolve_training_split_uses_explicit_toml_filepath(
    explicit_split_file_toml: Path, sample_split_data: dict
):
    """Explicit TOML split files load identically to JSON ones."""
    resolution = resolve_training_split(
        num_samples=100,
        test_ratio=0.2,
        val_ratio=0.2,
        seed=FIXED_SEED,
        persist_to=None,
        session_name="ignored",
        explicit_filepath=explicit_split_file_toml,
    )

    split = resolution.index_split
    assert list(split.train) == sample_split_data["train"]
    assert list(split.validation) == sample_split_data["validation"]
    assert list(split.test) == sample_split_data["test"]
    assert split.predict is not None
    assert list(split.predict) == sample_split_data["predict"]


def test_resolve_training_split_rejects_unsupported_extension(tmp_path: Path):
    """Split files with an unsupported extension fail fast with a clear error."""
    split_file = tmp_path / "custom_split.yaml"
    split_file.write_text("train: [0, 1]")

    with pytest.raises(ValueError, match="Unsupported split file format"):
        resolve_training_split(
            num_samples=50,
            test_ratio=0.2,
            val_ratio=0.2,
            seed=FIXED_SEED,
            persist_to=None,
            session_name="unsupported_test",
            explicit_filepath=split_file,
        )


def test_session_name_only_affects_artifact_naming_not_split_content():
    """Session name changes artifact naming only; the split content is seed-determined."""
    split1 = resolve_training_split(
        num_samples=50,
        test_ratio=0.2,
        val_ratio=0.2,
        seed=FIXED_SEED,
        persist_to=None,
        session_name="session_a",
    )
    split2 = resolve_training_split(
        num_samples=50,
        test_ratio=0.2,
        val_ratio=0.2,
        seed=FIXED_SEED,
        persist_to=None,
        session_name="session_b",
    )

    # Same seed -> identical split content regardless of session name.
    assert split1.index_split == split2.index_split
    # Session name only changes the artifact naming metadata.
    assert split1.artifact_filename != split2.artifact_filename

    split3 = resolve_training_split(
        num_samples=50,
        test_ratio=0.2,
        val_ratio=0.2,
        seed=OTHER_SEED,
        persist_to=None,
        session_name="session_a",
    )
    # Different seed -> different split content.
    assert split1.index_split != split3.index_split


def test_resolve_training_split_rejects_corrupt_explicit_file(tmp_path: Path):
    """Corrupt explicit split files fail fast instead of silently regenerating."""
    split_file = tmp_path / "corrupt_split.json"
    split_file.write_text("not valid json{{{")

    with pytest.raises(Exception):
        resolve_training_split(
            num_samples=50,
            test_ratio=0.2,
            val_ratio=0.2,
            seed=FIXED_SEED,
            persist_to=None,
            session_name="corrupt_test",
            explicit_filepath=split_file,
        )


def test_split_ratios_are_respected():
    """Test that split ratios are approximately correct."""
    num_samples = 1000
    test_ratio = 0.15
    val_ratio = 0.15

    split = resolve_training_split(
        num_samples=num_samples,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=FIXED_SEED,
        persist_to=None,
        session_name="ratio_test",
    ).index_split

    actual_test_ratio = len(split.test) / num_samples
    actual_val_ratio = len(split.validation) / num_samples
    actual_train_ratio = len(split.train) / num_samples

    assert abs(actual_test_ratio - test_ratio) < 0.01
    assert abs(actual_val_ratio - val_ratio) < 0.01
    assert abs(actual_train_ratio - (1 - test_ratio - val_ratio)) < 0.01


def test_split_indices_are_unique_and_complete():
    """Test that split indices don't overlap and cover all samples."""
    num_samples = 100
    split = resolve_training_split(
        num_samples=num_samples,
        test_ratio=0.2,
        val_ratio=0.2,
        seed=FIXED_SEED,
        persist_to=None,
        session_name="unique_test",
    ).index_split

    all_indices = set(split.train) | set(split.validation) | set(split.test)
    if split.predict:
        all_indices |= set(split.predict)

    assert len(all_indices) == num_samples
    assert all_indices == set(range(num_samples))
