"""Tests for DatasetBuilder.build_split_for_training / build_split_for_evaluation.

Covers the SRP split replacing the old shared build_split():
- Training path persists the resolved split locally under
  <default_root_dir>/splits/ when a root dir is configured, and leaves it
  in-memory only when it isn't.
- Evaluation path reloads an explicit filepath verbatim, or auto-resolves a
  unique colocated splits/ directory next to the checkpoint, and fails loud
  when that resolution is ambiguous or impossible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dlkit.common.errors import ConfigurationError
from dlkit.engine.workflows.factories.dataset_builder import DatasetBuilder
from dlkit.infrastructure.config.job_config import InferenceJobConfig, JobConfig, TrainingJobConfig
from dlkit.infrastructure.config.run_settings import RunSettings

NUM_SAMPLES: int = 30
FIXED_SEED: int = 99


@pytest.fixture
def dataset_builder() -> DatasetBuilder:
    """A fresh DatasetBuilder instance.

    Returns:
        DatasetBuilder: Builder under test.
    """
    return DatasetBuilder()


@pytest.fixture
def dummy_dataset() -> list[int]:
    """A Sized stand-in dataset of NUM_SAMPLES elements.

    Returns:
        list[int]: A list of NUM_SAMPLES elements.
    """
    return list(range(NUM_SAMPLES))


def _make_training_settings(
    *, default_root_dir: Path | None, seed: int | None = FIXED_SEED
) -> TrainingJobConfig:
    """Build a minimal TrainingJobConfig for split-resolution tests.

    Args:
        default_root_dir: Trainer's default_root_dir, or None to omit it.
        seed: run.seed override, or None to leave it unset.

    Returns:
        A validated TrainingJobConfig.
    """
    trainer_cfg: dict[str, object] = {"accelerator": "cpu"}
    if default_root_dir is not None:
        trainer_cfg["default_root_dir"] = str(default_root_dir)

    return TrainingJobConfig.model_validate(
        {
            "run": {"type": "train", "seed": seed},
            "experiment": {"name": "split_test"},
            "model": {"class": "Dummy"},
            "data": {"batch_size": 8, "num_workers": 0},
            "training": {"trainer": trainer_cfg},
        }
    )


def _write_split_file(path: Path, *, num_samples: int) -> Path:
    """Write a minimal valid split file covering num_samples indices.

    Args:
        path: Destination file path.
        num_samples: Total number of samples the split must cover.

    Returns:
        The path written to.
    """
    train_end = num_samples - 4
    val_end = num_samples - 2
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "train": list(range(train_end)),
                "validation": list(range(train_end, val_end)),
                "test": list(range(val_end, num_samples)),
            }
        )
    )
    return path


def test_build_split_for_training_persists_when_root_dir_configured(
    tmp_path: Path, dataset_builder: DatasetBuilder, dummy_dataset: list[int]
) -> None:
    """A configured default_root_dir causes the split to be written locally."""
    settings = _make_training_settings(default_root_dir=tmp_path)

    resolution = dataset_builder.build_split_for_training(settings, dummy_dataset)

    expected_path = tmp_path / "splits" / f"split_test_{NUM_SAMPLES}_split.json"
    assert expected_path.exists()
    assert (
        len(resolution.index_split.train)
        + len(resolution.index_split.validation)
        + len(resolution.index_split.test)
        == NUM_SAMPLES
    )


def test_build_split_for_training_stays_in_memory_without_root_dir(
    tmp_path: Path, dataset_builder: DatasetBuilder, dummy_dataset: list[int]
) -> None:
    """Without a configured default_root_dir, no split file is written anywhere."""
    settings = _make_training_settings(default_root_dir=None)

    dataset_builder.build_split_for_training(settings, dummy_dataset)

    assert not any(tmp_path.rglob("*.json"))


def test_build_split_for_training_is_deterministic_given_a_fixed_seed(
    dataset_builder: DatasetBuilder, dummy_dataset: list[int]
) -> None:
    """Same run.seed twice produces the identical resolved split."""
    settings = _make_training_settings(default_root_dir=None, seed=FIXED_SEED)

    first = dataset_builder.build_split_for_training(settings, dummy_dataset)
    second = dataset_builder.build_split_for_training(settings, dummy_dataset)

    assert first.index_split == second.index_split


def test_build_split_for_evaluation_uses_explicit_filepath(
    tmp_path: Path, dataset_builder: DatasetBuilder, dummy_dataset: list[int]
) -> None:
    """An explicit data.splits.filepath is reloaded verbatim."""
    split_file = _write_split_file(tmp_path / "explicit_split.json", num_samples=NUM_SAMPLES)
    settings = InferenceJobConfig.model_validate(
        {
            "run": {"type": "predict"},
            "model": {"class": "Dummy", "checkpoint": "unused.ckpt"},
            "data": {
                "batch_size": 8,
                "num_workers": 0,
                "splits": {"filepath": str(split_file)},
            },
        }
    )

    resolution = dataset_builder.build_split_for_evaluation(settings, dummy_dataset)

    assert resolution.source_path == split_file


def test_build_split_for_evaluation_auto_resolves_colocated_split(
    tmp_path: Path, dataset_builder: DatasetBuilder, dummy_dataset: list[int]
) -> None:
    """A unique splits/*.json sibling of the checkpoint's run root is auto-located."""
    run_root = tmp_path / "run"
    checkpoint = run_root / "checkpoints" / "best.ckpt"
    split_file = _write_split_file(
        run_root / "splits" / "session_split.json", num_samples=NUM_SAMPLES
    )
    settings = InferenceJobConfig.model_validate(
        {
            "run": {"type": "predict"},
            "model": {"class": "Dummy", "checkpoint": str(checkpoint)},
            "data": {"batch_size": 8, "num_workers": 0},
        }
    )

    resolution = dataset_builder.build_split_for_evaluation(settings, dummy_dataset)

    assert resolution.source_path == split_file


def test_build_split_for_evaluation_raises_on_ambiguous_colocated_splits(
    tmp_path: Path, dataset_builder: DatasetBuilder, dummy_dataset: list[int]
) -> None:
    """More than one candidate splits/*.json file fails loud rather than guessing."""
    run_root = tmp_path / "run"
    checkpoint = run_root / "checkpoints" / "best.ckpt"
    _write_split_file(run_root / "splits" / "a_split.json", num_samples=NUM_SAMPLES)
    _write_split_file(run_root / "splits" / "b_split.json", num_samples=NUM_SAMPLES)
    settings = InferenceJobConfig.model_validate(
        {
            "run": {"type": "predict"},
            "model": {"class": "Dummy", "checkpoint": str(checkpoint)},
            "data": {"batch_size": 8, "num_workers": 0},
        }
    )

    with pytest.raises(ConfigurationError, match="Set data.splits.filepath explicitly"):
        dataset_builder.build_split_for_evaluation(settings, dummy_dataset)


def test_build_split_for_evaluation_raises_when_no_checkpoint_and_no_filepath(
    dataset_builder: DatasetBuilder, dummy_dataset: list[int]
) -> None:
    """No explicit filepath and no model configured at all fails loud.

    Uses a bare JobConfig (model=None) rather than InferenceJobConfig, since
    InferenceJobConfig's own validator already guarantees a checkpoint is
    set — this exercises build_split_for_evaluation's own defensive guard
    for the general JobConfig type it's typed to accept.
    """
    settings = JobConfig(run=RunSettings(type="predict"))

    with pytest.raises(ConfigurationError, match="no model.checkpoint is configured"):
        dataset_builder.build_split_for_evaluation(settings, dummy_dataset)
