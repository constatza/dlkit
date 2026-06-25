"""Smoke tests: engine-facing config types load and wire cleanly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlkit.infrastructure.config.job_config import TrainingJobConfig


@pytest.fixture
def npy_data_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal npy files for entry validation."""
    x_path = tmp_path / "x.npy"
    y_path = tmp_path / "y.npy"
    np.save(str(x_path), np.zeros((10, 2)))
    np.save(str(y_path), np.zeros((10, 1)))
    return x_path, y_path


@pytest.fixture
def training_job_config(npy_data_paths: tuple[Path, Path]) -> TrainingJobConfig:
    """Minimal valid TrainingJobConfig with real file paths."""
    x_path, y_path = npy_data_paths
    return TrainingJobConfig.model_validate(
        {
            "run": {"type": "train", "seed": 1},
            "model": {"class": "ConstantWidthFFNN", "module_path": "dlkit.domain.nn"},
            "data": {
                "class": "FlexibleDataset",
                "features": [{"name": "x", "format": "npy", "path": str(x_path)}],
                "targets": [{"name": "y", "format": "npy", "path": str(y_path)}],
            },
            "training": {"loss": "mse"},
        }
    )


def test_training_job_config_is_importable(training_job_config: TrainingJobConfig) -> None:
    """Smoke test: engine-facing config type loads cleanly."""
    assert training_job_config.run.seed == 1
    assert training_job_config.model.name == "ConstantWidthFFNN"
    assert training_job_config.training.loss.name == "mse"


def test_training_job_config_data_fields(training_job_config: TrainingJobConfig) -> None:
    """Data fields are accessible at job.data (not job.DATASET / job.DATAMODULE)."""
    assert training_job_config.data.batch_size == 64
    assert training_job_config.data.num_workers == 0
    assert training_job_config.data.splits is not None


def test_training_job_config_training_fields(training_job_config: TrainingJobConfig) -> None:
    """Training fields are accessible at job.training (not job.TRAINING)."""
    assert training_job_config.training.trainer is not None
    assert training_job_config.training.stopping.patience == 10
    assert training_job_config.training.stopping.monitor == "val/loss"


def test_training_job_config_tracking_defaults(training_job_config: TrainingJobConfig) -> None:
    """Tracking section defaults to 'none' backend when not specified."""
    assert training_job_config.tracking.backend == "none"
    assert training_job_config.experiment is None
