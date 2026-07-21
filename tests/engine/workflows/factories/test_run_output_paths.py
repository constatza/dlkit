"""Tests for resolve_local_artifact_root() — the single source of truth for
a run's local file-based artifact output root."""

from __future__ import annotations

from pathlib import Path

from dlkit.engine.workflows.factories.run_output_paths import resolve_local_artifact_root
from dlkit.infrastructure.config.job_config import JobConfig, TrainingJobConfig
from dlkit.infrastructure.config.run_settings import RunSettings


def _make_training_settings(*, default_root_dir: Path | None) -> TrainingJobConfig:
    """Build a minimal TrainingJobConfig with an optional trainer default_root_dir.

    Args:
        default_root_dir: Trainer's default_root_dir, or None to omit it.

    Returns:
        A validated TrainingJobConfig.
    """
    trainer_cfg: dict[str, object] = {"accelerator": "cpu"}
    if default_root_dir is not None:
        trainer_cfg["default_root_dir"] = str(default_root_dir)

    return TrainingJobConfig.model_validate(
        {
            "run": {"type": "train"},
            "model": {"class": "Dummy"},
            "data": {"batch_size": 8, "num_workers": 0},
            "training": {"trainer": trainer_cfg},
        }
    )


def test_resolve_local_artifact_root_returns_configured_root(tmp_path: Path) -> None:
    """A configured trainer.default_root_dir is returned as a Path."""
    settings = _make_training_settings(default_root_dir=tmp_path)

    root = resolve_local_artifact_root(settings)

    assert root == tmp_path


def test_resolve_local_artifact_root_returns_none_without_default_root_dir() -> None:
    """No configured default_root_dir resolves to None."""
    settings = _make_training_settings(default_root_dir=None)

    assert resolve_local_artifact_root(settings) is None


def test_resolve_local_artifact_root_returns_none_without_training_section() -> None:
    """A JobConfig with no training section at all resolves to None."""
    settings = JobConfig(run=RunSettings(type="predict"))

    assert resolve_local_artifact_root(settings) is None


def test_resolve_local_artifact_root_returns_none_with_default_trainer_section() -> None:
    """A trainer sub-section with defaults (no default_root_dir) resolves to None."""
    settings = TrainingJobConfig.model_validate(
        {
            "run": {"type": "train"},
            "model": {"class": "Dummy"},
            "data": {"batch_size": 8, "num_workers": 0},
            "training": {},
        }
    )

    assert resolve_local_artifact_root(settings) is None
