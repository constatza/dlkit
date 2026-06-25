"""Tests for MLflowTracker dataset lineage logging."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dlkit.common.hooks import ParamValue
from dlkit.engine.tracking.dataset_logger import DatasetLogger
from dlkit.engine.tracking.interfaces import IRunContext
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.data_settings import DataSettings
from dlkit.infrastructure.config.entry_types import NpyEntry
from dlkit.infrastructure.config.enums import DatasetFamily
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.run_settings import RunSettings


class _DatasetRunContext(IRunContext):
    """Run context that records dataset and artifact lineage calls."""

    def __init__(self):
        self._run_id = "dataset-run"
        self.logged_datasets: list[dict[str, Any]] = []
        self.manifests: list[dict[str, Any]] = []
        self.tags: dict[str, str] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def experiment_id(self) -> str | None:
        return "dataset-experiment"

    @property
    def tracking_uri(self) -> str | None:
        return "sqlite:///tmp/mlflow.db"

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        pass

    def log_artifact_content(self, content: str | bytes, artifact_file: str) -> None:
        if artifact_file.startswith("lineage/"):
            payload = content.decode("utf-8") if isinstance(content, bytes) else content
            self.manifests.append(json.loads(payload))

    def log_artifact(self, artifact_path: Path, artifact_dir: str = "") -> None:
        pass

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_dataset(
        self,
        dataset: Any,
        context: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.logged_datasets.append({"dataset": dataset, "context": context, "tags": tags})

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        *,
        registered_model_name: str | None = None,
        signature: Any | None = None,
        input_example: Any | None = None,
    ) -> str | None:
        return None

    def get_latest_model_version(
        self,
        model_name: str,
        *,
        run_id: str | None = None,
        artifact_path: str | None = None,
    ) -> int | None:
        return None

    def set_model_alias(self, model_name: str, alias: str, version: int) -> None:
        pass

    def set_model_version_tag(
        self,
        model_name: str,
        version: int,
        key: str,
        value: str,
    ) -> None:
        pass


@pytest.fixture
def feature_npy(tmp_path: Path) -> Path:
    """Create a feature .npy file fixture.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to the created .npy file.
    """
    path = tmp_path / "features.npy"
    np.save(path, np.array([[1.0], [2.0]], dtype=np.float32))
    return path


@pytest.fixture
def target_npy(tmp_path: Path) -> Path:
    """Create a target .npy file fixture.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to the created .npy file.
    """
    path = tmp_path / "targets.npy"
    np.save(path, np.array([[0.0], [1.0]], dtype=np.float32))
    return path


@pytest.fixture
def job_with_entries(feature_npy: Path, target_npy: Path) -> JobConfig:
    """JobConfig with feature and target NpyEntries.

    Args:
        feature_npy: Feature numpy file path fixture.
        target_npy: Target numpy file path fixture.

    Returns:
        JobConfig with data section containing feature and target entries.
    """
    return JobConfig(
        run=RunSettings(type="train"),
        data=DataSettings(
            name="CustomDataset",
            features=(NpyEntry(name="x", path=feature_npy, data_role=DataRole.FEATURE),),
            targets=(NpyEntry(name="y", path=target_npy, data_role=DataRole.TARGET),),
        ),
    )


@pytest.fixture
def job_with_empty_data() -> JobConfig:
    """JobConfig with an empty DataSettings (no features or targets).

    Returns:
        JobConfig with data section containing no entries.
    """
    return JobConfig(
        run=RunSettings(type="train"),
        data=DataSettings(name="CustomDataset"),
    )


def test_logs_structured_entry_dataset_for_unsupported_runtime_dataset(
    job_with_entries: JobConfig,
) -> None:
    run_context = _DatasetRunContext()
    tracker = DatasetLogger()

    unsupported_dataset = SimpleNamespace()
    datamodule = SimpleNamespace(dataset=unsupported_dataset)

    tracker.log_dataset_to_run(datamodule, run_context, job_with_entries)

    assert len(run_context.logged_datasets) == 1
    assert run_context.logged_datasets[0]["context"] == "training"
    assert len(run_context.manifests) == 1
    manifest = run_context.manifests[0]
    assert manifest["dataset_name"] == "CustomDataset"
    assert manifest["structured_mlflow_dataset_logged"] is True
    assert manifest["source_count"] == 2
    assert run_context.tags["dataset_source_count"] == "2"
    assert "dataset_fingerprint" in run_context.tags


def test_logs_manifest_only_for_empty_sources_and_unsupported_dataset(
    job_with_empty_data: JobConfig,
) -> None:
    run_context = _DatasetRunContext()
    tracker = DatasetLogger()

    datamodule = SimpleNamespace(dataset=SimpleNamespace())

    tracker.log_dataset_to_run(datamodule, run_context, job_with_empty_data)

    assert run_context.logged_datasets == []
    assert len(run_context.manifests) == 1
    manifest = run_context.manifests[0]
    assert manifest["structured_mlflow_dataset_logged"] is False
    assert manifest["source_count"] == 0
    assert run_context.tags["dataset_source_count"] == "0"


def test_collects_graph_sources_from_dataset_settings_fields(tmp_path: Path) -> None:
    x_path = tmp_path / "x.npy"
    edge_path = tmp_path / "edge.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.array([[1.0], [2.0]], dtype=np.float32))
    np.save(edge_path, np.array([[0, 1], [1, 0]], dtype=np.int64))
    np.save(y_path, np.array([[0.0], [1.0]], dtype=np.float32))

    job = JobConfig(
        run=RunSettings(type="train"),
        data=DataSettings(
            name="GraphDataset",
            family=DatasetFamily.GRAPH,
            features=(
                NpyEntry(name="x", path=x_path, data_role=DataRole.FEATURE),
                NpyEntry(name="edge_index", path=edge_path, data_role=DataRole.FEATURE),
            ),
            targets=(NpyEntry(name="y", path=y_path, data_role=DataRole.TARGET),),
        ),
    )
    run_context = _DatasetRunContext()
    tracker = DatasetLogger()

    datamodule = SimpleNamespace(dataset=SimpleNamespace())
    tracker.log_dataset_to_run(datamodule, run_context, job)

    assert len(run_context.manifests) == 1
    manifest = run_context.manifests[0]
    assert manifest["source_count"] == 3
    assert run_context.tags["dataset_source_count"] == "3"


def test_logs_structured_tabular_dataset_when_dataframe_available(
    job_with_empty_data: JobConfig,
) -> None:
    run_context = _DatasetRunContext()
    tracker = DatasetLogger()

    tabular_dataset = SimpleNamespace(df=pd.DataFrame({"x": [1, 2], "y": [3, 4]}))
    datamodule = SimpleNamespace(dataset=tabular_dataset)

    tracker.log_dataset_to_run(datamodule, run_context, job_with_empty_data)

    assert len(run_context.logged_datasets) == 1
    assert run_context.logged_datasets[0]["context"] == "training"
    assert len(run_context.manifests) == 1
