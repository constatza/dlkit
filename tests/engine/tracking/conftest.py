"""Shared fixtures for `run_queries.py` and `checkpoint_recovery.py` tests.

Fixtures fall into two families:

- Pure-MLflow-client fixtures (`tracking_uri`, `mlflow_client`,
  `experiment_name`, `experiment_id`) for `test_run_queries.py`, which only
  needs runs to exist — not real training.
- Real-training fixtures (`default_checkpoint_run`,
  `custom_filename_checkpoint_run`, `no_checkpoint_run`,
  `multiple_checkpoint_files_with_best_run`,
  `multiple_checkpoint_files_without_best_run`) for
  `test_checkpoint_recovery.py`, which needs actual `checkpoints/*.ckpt`
  artifacts logged to a real (local sqlite) MLflow backend via
  `api_train()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient

from dlkit.common import TrainingResult
from dlkit.engine.tracking.mlflow_client_factory import MLflowClientFactory
from dlkit.infrastructure.config.job_config import TrainingJobConfig
from dlkit.interfaces.api.functions import train as api_train

FEATURE_SIZE = 4
TARGET_SIZE = 2
NUM_SAMPLES = 20
BATCH_SIZE = 4
EPOCHS = 1


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    """Isolated sqlite MLflow tracking URI, pinned as the active fluent URI.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        A `sqlite:///` tracking URI scoped to this test's `tmp_path`, already
        set as MLflow's active tracking URI (needed by fluent-API calls such
        as `mlflow.start_run`).
    """
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    return uri


@pytest.fixture
def mlflow_client(tracking_uri: str) -> MlflowClient:
    """MlflowClient bound to the isolated test tracking URI.

    Args:
        tracking_uri: Isolated sqlite tracking URI fixture.

    Returns:
        Configured MlflowClient instance.
    """
    return MLflowClientFactory.create_client(tracking_uri)


@pytest.fixture
def experiment_name() -> str:
    """Fixed experiment name used by the `run_queries` tests.

    Returns:
        A stable experiment name string.
    """
    return "run_queries_test_experiment"


@pytest.fixture
def experiment_id(mlflow_client: MlflowClient, experiment_name: str) -> str:
    """Create the test experiment and return its id.

    Args:
        mlflow_client: MlflowClient fixture bound to the test tracking URI.
        experiment_name: Fixed experiment name fixture.

    Returns:
        The created experiment's id.
    """
    return mlflow_client.create_experiment(experiment_name)


@pytest.fixture
def checkpoint_dataset(tmp_path: Path) -> dict[str, Path]:
    """Small synthetic supervised dataset for real (non `fast_dev_run`) training.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Mapping of `"features"`/`"targets"` to their `.npy` file paths.
    """
    rng = np.random.default_rng(42)
    features = rng.standard_normal((NUM_SAMPLES, FEATURE_SIZE)).astype(np.float32)
    targets = rng.integers(0, TARGET_SIZE, size=(NUM_SAMPLES, 1)).astype(np.float32)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    features_path = data_dir / "features.npy"
    targets_path = data_dir / "targets.npy"
    np.save(features_path, features)
    np.save(targets_path, targets)
    return {"features": features_path, "targets": targets_path}


def _build_training_job_config(
    *,
    feature_path: Path,
    target_path: Path,
    tracking_uri: str,
    default_root_dir: Path,
    experiment_name: str,
    enable_checkpointing: bool = True,
    callbacks: list[dict[str, Any]] | None = None,
) -> TrainingJobConfig:
    """Build a TrainingJobConfig for a real, MLflow-tracked training run.

    Args:
        feature_path: Path to the feature `.npy` file.
        target_path: Path to the target `.npy` file.
        tracking_uri: MLflow tracking URI to record the run under.
        default_root_dir: Local root directory for trainer outputs; created
            if it does not already exist (required because `TrainerSettings`
            validates it as a `DirectoryPath`).
        experiment_name: MLflow experiment name for the run.
        enable_checkpointing: Whether Lightning checkpointing is enabled.
            When `True` and `callbacks` is `None`, dlkit auto-injects its
            default best-only `ModelCheckpoint` callback.
        callbacks: Optional explicit callback overrides (e.g. a
            `ModelCheckpoint` with a custom `filename=` template). `None`
            uses dlkit's auto-injected default.

    Returns:
        TrainingJobConfig ready for `api_train()`.
    """
    default_root_dir.mkdir(parents=True, exist_ok=True)

    trainer_dict: dict[str, Any] = {
        "fast_dev_run": False,
        "enable_checkpointing": enable_checkpointing,
        "accelerator": "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "max_epochs": EPOCHS,
        "default_root_dir": str(default_root_dir),
    }
    if callbacks is not None:
        trainer_dict["callbacks"] = callbacks

    payload: dict[str, Any] = {
        "run": {"type": "train", "seed": 42},
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
            "trainer": trainer_dict,
            "optimizer": {"name": "AdamW", "lr": 1e-3},
            "metrics": [{"name": "MeanSquaredError", "module_path": "dlkit.domain.metrics"}],
        },
        "tracking": {"backend": "mlflow", "uri": tracking_uri},
    }
    return TrainingJobConfig.model_validate(payload)


@pytest.fixture
def default_checkpoint_run(
    checkpoint_dataset: dict[str, Path], tracking_uri: str, tmp_path: Path
) -> TrainingResult:
    """TrainingResult for a run trained with dlkit's default (best-only) checkpoint config.

    Args:
        checkpoint_dataset: Synthetic dataset fixture.
        tracking_uri: Isolated sqlite tracking URI fixture.
        tmp_path: Pytest temporary directory fixture.

    Returns:
        TrainingResult with `mlflow_run_id` set, logging only `best.ckpt`.
    """
    config = _build_training_job_config(
        feature_path=checkpoint_dataset["features"],
        target_path=checkpoint_dataset["targets"],
        tracking_uri=tracking_uri,
        default_root_dir=tmp_path / "default_checkpoint_output",
        experiment_name="checkpoint_recovery_default",
    )
    return api_train(config)


@pytest.fixture
def custom_filename_checkpoint_run(
    checkpoint_dataset: dict[str, Path], tracking_uri: str, tmp_path: Path
) -> TrainingResult:
    """TrainingResult for a run trained with a custom `ModelCheckpoint` filename template.

    Proves `download_checkpoint_artifact` is name-agnostic: it must still
    find and download the single checkpoint file even though it isn't named
    `best.ckpt`.

    Args:
        checkpoint_dataset: Synthetic dataset fixture.
        tracking_uri: Isolated sqlite tracking URI fixture.
        tmp_path: Pytest temporary directory fixture.

    Returns:
        TrainingResult with `mlflow_run_id` set, logging a single checkpoint
        file named `my-custom-name.ckpt`.
    """
    config = _build_training_job_config(
        feature_path=checkpoint_dataset["features"],
        target_path=checkpoint_dataset["targets"],
        tracking_uri=tracking_uri,
        default_root_dir=tmp_path / "custom_filename_checkpoint_output",
        experiment_name="checkpoint_recovery_custom_filename",
        callbacks=[
            {
                "name": "ModelCheckpoint",
                "monitor": "val/loss",
                "mode": "min",
                "save_top_k": 1,
                "filename": "my-custom-name",
            }
        ],
    )
    return api_train(config)


@pytest.fixture
def no_checkpoint_run(
    checkpoint_dataset: dict[str, Path], tracking_uri: str, tmp_path: Path
) -> TrainingResult:
    """TrainingResult for a tracked run with checkpointing disabled entirely.

    Args:
        checkpoint_dataset: Synthetic dataset fixture.
        tracking_uri: Isolated sqlite tracking URI fixture.
        tmp_path: Pytest temporary directory fixture.

    Returns:
        TrainingResult with `mlflow_run_id` set, but no `checkpoints/` artifacts.
    """
    config = _build_training_job_config(
        feature_path=checkpoint_dataset["features"],
        target_path=checkpoint_dataset["targets"],
        tracking_uri=tracking_uri,
        default_root_dir=tmp_path / "no_checkpoint_output",
        experiment_name="checkpoint_recovery_none",
        enable_checkpointing=False,
    )
    return api_train(config)


@pytest.fixture
def multiple_checkpoint_files_with_best_run(
    default_checkpoint_run: TrainingResult, tracking_uri: str, tmp_path: Path
) -> TrainingResult:
    """A run with two files under `checkpoints/`, one of them `best.ckpt`.

    Built on top of `default_checkpoint_run` (already has one real
    `checkpoints/best.ckpt` artifact, since the default config's `filename`
    is `"best"`), then a second, independent checkpoint file is uploaded
    directly via the MLflow client — not by relying on training to ever
    produce two files, since dlkit's own checkpoint config never does.
    Exercises the `best.ckpt`-disambiguation success path.

    Args:
        default_checkpoint_run: TrainingResult with one `best.ckpt`
            artifact already logged.
        tracking_uri: Isolated sqlite tracking URI fixture.
        tmp_path: Pytest temporary directory fixture.

    Returns:
        TrainingResult whose run now has two files under `checkpoints/`,
        including `best.ckpt`.
    """
    assert default_checkpoint_run.mlflow_run_id is not None
    extra_ckpt = tmp_path / "extra.ckpt"
    extra_ckpt.write_bytes(b"not a real checkpoint, just a second file")

    client = MLflowClientFactory.create_client(tracking_uri)
    client.log_artifact(
        default_checkpoint_run.mlflow_run_id, str(extra_ckpt), artifact_path="checkpoints"
    )
    return default_checkpoint_run


@pytest.fixture
def multiple_checkpoint_files_without_best_run(
    custom_filename_checkpoint_run: TrainingResult, tracking_uri: str, tmp_path: Path
) -> TrainingResult:
    """A run with two files under `checkpoints/`, neither of them `best.ckpt`.

    Built on top of `custom_filename_checkpoint_run` (already has one real
    `checkpoints/my-custom-name.ckpt` artifact), then a second, independent,
    non-`best.ckpt`-named file is uploaded directly via the MLflow client.
    Exercises the unresolvable-ambiguity error path.

    Args:
        custom_filename_checkpoint_run: TrainingResult with one
            custom-named checkpoint artifact already logged.
        tracking_uri: Isolated sqlite tracking URI fixture.
        tmp_path: Pytest temporary directory fixture.

    Returns:
        TrainingResult whose run now has two files under `checkpoints/`,
        neither named `best.ckpt`.
    """
    assert custom_filename_checkpoint_run.mlflow_run_id is not None
    extra_ckpt = tmp_path / "another-extra.ckpt"
    extra_ckpt.write_bytes(b"not a real checkpoint, just a second file")

    client = MLflowClientFactory.create_client(tracking_uri)
    client.log_artifact(
        custom_filename_checkpoint_run.mlflow_run_id, str(extra_ckpt), artifact_path="checkpoints"
    )
    return custom_filename_checkpoint_run
