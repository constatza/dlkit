"""Shared fixtures for runtime workflow entrypoint tests.

Builds tiny synthetic datasets and real (sqlite-backed) MLflow tracking
config, mirroring `tests/integration/conftest.py`'s `_make_training_job_config`
and `tests/integration/test_evaluate_multirun_integration.py`'s
`sweep_variant_settings` — replicated locally so these tests stay independent
of `tests/integration/`'s fixture set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tomli_w

from dlkit.infrastructure.config.job_config import SearchJobConfig, TrainingJobConfig

FEATURE_SIZE = 4
TARGET_SIZE = 1
NUM_SAMPLES = 20
BATCH_SIZE = 4


@pytest.fixture
def minimal_dataset(tmp_path: Path) -> dict[str, Path]:
    """Create a tiny synthetic supervised dataset for fast entrypoint tests.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Dict with "features" and "targets" keys pointing at .npy files.
    """
    rng = np.random.default_rng(42)
    features = rng.standard_normal((NUM_SAMPLES, FEATURE_SIZE)).astype(np.float32)
    targets = rng.standard_normal((NUM_SAMPLES, TARGET_SIZE)).astype(np.float32)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    feature_path = data_dir / "features.npy"
    target_path = data_dir / "targets.npy"
    np.save(feature_path, features)
    np.save(target_path, targets)
    return {"features": feature_path, "targets": target_path}


@pytest.fixture
def sqlite_tracking_uri(tmp_path: Path) -> str:
    """A real, file-backed sqlite MLflow tracking URI under tmp_path.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        A `sqlite:///...` tracking URI string.
    """
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}"


def _tracking_dict(tracking_uri: str) -> dict[str, Any]:
    """Build a `[tracking]` payload pointed at a real mlflow backend.

    Args:
        tracking_uri: Sqlite tracking URI to embed.

    Returns:
        Dict payload for the `tracking` section of a job config.
    """
    return {"backend": "mlflow", "uri": tracking_uri}


def _train_payload(
    *,
    feature_path: Path,
    target_path: Path,
    tracking_uri: str,
    experiment_name: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a minimal TrainingJobConfig payload.

    Args:
        feature_path: Path to the feature .npy file.
        target_path: Path to the target .npy file.
        tracking_uri: Sqlite tracking URI shared across the sweep.
        experiment_name: MLflow experiment name for this child (patched away
            by the orchestrator anyway, but required by JobConfig).
        seed: Global random seed for this child.

    Returns:
        Dict payload suitable for `TrainingJobConfig.model_validate()` or
        `tomli_w.dump()`.
    """
    return {
        "run": {"type": "train", "seed": seed},
        "experiment": {"name": experiment_name},
        "model": {
            "class": "FFNN",
            "module_path": "dlkit.domain.nn",
            "hidden_size": FEATURE_SIZE,
            "num_layers": 0,
        },
        "data": {
            "class": "FlexibleDataset",
            "module_path": "dlkit.engine.data.datasets",
            "batch_size": BATCH_SIZE,
            "num_workers": 0,
            "shuffle": True,
            "pin_memory": False,
            "persistent_workers": False,
            "features": [{"name": "x", "path": str(feature_path), "format": "npy"}],
            "targets": [{"name": "y", "path": str(target_path), "format": "npy"}],
        },
        "training": {
            "loss": "mse",
            "trainer": {
                "fast_dev_run": True,
                "enable_checkpointing": False,
                "accelerator": "cpu",
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "max_epochs": 1,
            },
            "optimizer": {"name": "AdamW", "lr": 1e-3},
        },
        "tracking": _tracking_dict(tracking_uri),
    }


def _search_payload(
    *,
    feature_path: Path,
    target_path: Path,
    tracking_uri: str,
    experiment_name: str,
) -> dict[str, Any]:
    """Build a minimal SearchJobConfig payload (1-trial HPO study).

    Args:
        feature_path: Path to the feature .npy file.
        target_path: Path to the target .npy file.
        tracking_uri: Sqlite tracking URI shared across the sweep.
        experiment_name: MLflow experiment name for this child.

    Returns:
        Dict payload suitable for `SearchJobConfig.model_validate()` or
        `tomli_w.dump()`.
    """
    payload = _train_payload(
        feature_path=feature_path,
        target_path=target_path,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    payload["run"]["type"] = "search"
    payload["search"] = {
        "n_trials": 1,
        "direction": "minimize",
        "objective": "val/loss",
        "space": {"model.hidden_size": {"type": "categorical", "choices": [FEATURE_SIZE]}},
    }
    return payload


@pytest.fixture
def train_job_configs(
    minimal_dataset: dict[str, Path], sqlite_tracking_uri: str
) -> tuple[TrainingJobConfig, TrainingJobConfig]:
    """Two real TrainingJobConfig objects sharing a sqlite tracking backend.

    Args:
        minimal_dataset: Dataset path fixture.
        sqlite_tracking_uri: Shared sqlite tracking URI fixture.

    Returns:
        Tuple of two validated TrainingJobConfig instances.
    """

    def _build(index: int) -> TrainingJobConfig:
        return TrainingJobConfig.model_validate(
            _train_payload(
                feature_path=minimal_dataset["features"],
                target_path=minimal_dataset["targets"],
                tracking_uri=sqlite_tracking_uri,
                experiment_name=f"child-{index}",
                seed=index,
            )
        )

    return _build(0), _build(1)


@pytest.fixture
def train_child_paths(
    minimal_dataset: dict[str, Path], sqlite_tracking_uri: str, tmp_path: Path
) -> tuple[Path, Path]:
    """Two child TrainingJobConfig TOML files on disk, for config-driven tests.

    Args:
        minimal_dataset: Dataset path fixture.
        sqlite_tracking_uri: Shared sqlite tracking URI fixture.
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Tuple of two absolute paths to written TOML files.
    """
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(2):
        path = jobs_dir / f"train_child_{i}.toml"
        payload = _train_payload(
            feature_path=minimal_dataset["features"],
            target_path=minimal_dataset["targets"],
            tracking_uri=sqlite_tracking_uri,
            experiment_name=f"child-{i}",
            seed=i,
        )
        path.write_bytes(tomli_w.dumps(payload).encode("utf-8"))
        paths.append(path)
    return paths[0], paths[1]


@pytest.fixture
def train_and_search_child_paths(
    minimal_dataset: dict[str, Path], sqlite_tracking_uri: str, tmp_path: Path
) -> tuple[Path, Path]:
    """One train-child and one search-child TOML file on disk.

    Args:
        minimal_dataset: Dataset path fixture.
        sqlite_tracking_uri: Shared sqlite tracking URI fixture.
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Tuple of (train_child_path, search_child_path).
    """
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    train_path = jobs_dir / "train_child.toml"
    train_payload = _train_payload(
        feature_path=minimal_dataset["features"],
        target_path=minimal_dataset["targets"],
        tracking_uri=sqlite_tracking_uri,
        experiment_name="train-child",
    )
    train_path.write_bytes(tomli_w.dumps(train_payload).encode("utf-8"))

    search_path = jobs_dir / "search_child.toml"
    search_payload = _search_payload(
        feature_path=minimal_dataset["features"],
        target_path=minimal_dataset["targets"],
        tracking_uri=sqlite_tracking_uri,
        experiment_name="search-child",
    )
    search_path.write_bytes(tomli_w.dumps(search_payload).encode("utf-8"))

    return train_path, search_path


@pytest.fixture
def search_job_config(
    minimal_dataset: dict[str, Path], sqlite_tracking_uri: str
) -> SearchJobConfig:
    """A real, minimal SearchJobConfig (1-trial HPO study).

    Args:
        minimal_dataset: Dataset path fixture.
        sqlite_tracking_uri: Shared sqlite tracking URI fixture.

    Returns:
        Validated SearchJobConfig.
    """
    return SearchJobConfig.model_validate(
        _search_payload(
            feature_path=minimal_dataset["features"],
            target_path=minimal_dataset["targets"],
            tracking_uri=sqlite_tracking_uri,
            experiment_name="search-child",
        )
    )
