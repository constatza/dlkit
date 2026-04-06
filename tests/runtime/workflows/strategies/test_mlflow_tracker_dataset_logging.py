"""Tests for MLflowTracker dataset lineage logging."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from dlkit.engine.tracking.interfaces import IRunContext
from dlkit.engine.tracking.mlflow_tracker import MLflowTracker
from dlkit.infrastructure.config.data_entries import Feature, Target
from dlkit.infrastructure.config.dataset_settings import DatasetSettings
from dlkit.infrastructure.config.enums import DatasetFamily
from dlkit.infrastructure.config.general_settings import GeneralSettings


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

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
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


def _make_settings(tmp_path: Path) -> GeneralSettings:
    feature_path = tmp_path / "features.npy"
    target_path = tmp_path / "targets.npy"
    np.save(feature_path, np.array([[1.0], [2.0]], dtype=np.float32))
    np.save(target_path, np.array([[0.0], [1.0]], dtype=np.float32))

    dataset = DatasetSettings(
        name="CustomDataset",
        features=(Feature(name="x", path=feature_path),),
        targets=(Target(name="y", path=target_path),),
    )
    return GeneralSettings(DATASET=dataset)


def test_logs_structured_entry_dataset_for_unsupported_runtime_dataset(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    run_context = _DatasetRunContext()
    tracker = MLflowTracker(disable_autostart=True)

    unsupported_dataset = SimpleNamespace()
    datamodule = SimpleNamespace(dataset=unsupported_dataset)

    tracker.log_dataset_to_run(datamodule, run_context, settings)

    assert len(run_context.logged_datasets) == 1
    assert run_context.logged_datasets[0]["context"] == "training"
    assert len(run_context.manifests) == 1
    manifest = run_context.manifests[0]
    assert manifest["dataset_name"] == "CustomDataset"
    assert manifest["structured_mlflow_dataset_logged"] is True
    assert manifest["source_count"] == 2
    assert run_context.tags["dataset_source_count"] == "2"
    assert "dataset_fingerprint" in run_context.tags


def test_logs_manifest_only_for_empty_sources_and_unsupported_dataset(tmp_path: Path) -> None:
    settings = GeneralSettings(DATASET=DatasetSettings(name="CustomDataset"))
    run_context = _DatasetRunContext()
    tracker = MLflowTracker(disable_autostart=True)

    datamodule = SimpleNamespace(dataset=SimpleNamespace())

    tracker.log_dataset_to_run(datamodule, run_context, settings)

    assert run_context.logged_datasets == []
    assert len(run_context.manifests) == 1
    manifest = run_context.manifests[0]
    assert manifest["structured_mlflow_dataset_logged"] is False
    assert manifest["source_count"] == 0
    assert run_context.tags["dataset_source_count"] == "0"


def test_collects_graph_sources_from_dataset_settings_fields(tmp_path: Path) -> None:
    x = tmp_path / "x.npy"
    edge_index = tmp_path / "edge.npy"
    y = tmp_path / "y.npy"
    np.save(x, np.array([[1.0], [2.0]], dtype=np.float32))
    np.save(edge_index, np.array([[0, 1], [1, 0]], dtype=np.int64))
    np.save(y, np.array([[0.0], [1.0]], dtype=np.float32))

    settings = GeneralSettings(
        DATASET=DatasetSettings.model_validate(
            {
                "name": "GraphDataset",
                "type": DatasetFamily.GRAPH,
                "x": x,
                "edge_index": edge_index,
                "y": y,
            }
        )
    )
    run_context = _DatasetRunContext()
    tracker = MLflowTracker(disable_autostart=True)

    datamodule = SimpleNamespace(dataset=SimpleNamespace())
    tracker.log_dataset_to_run(datamodule, run_context, settings)

    assert len(run_context.manifests) == 1
    manifest = run_context.manifests[0]
    assert manifest["source_count"] == 3
    assert run_context.tags["dataset_source_count"] == "3"


def test_logs_structured_tabular_dataset_when_dataframe_available(tmp_path: Path) -> None:
    settings = GeneralSettings(DATASET=DatasetSettings(name="CustomDataset"))
    run_context = _DatasetRunContext()
    tracker = MLflowTracker(disable_autostart=True)

    tabular_dataset = SimpleNamespace(df=pd.DataFrame({"x": [1, 2], "y": [3, 4]}))
    datamodule = SimpleNamespace(dataset=tabular_dataset)

    tracker.log_dataset_to_run(datamodule, run_context, settings)

    assert len(run_context.logged_datasets) == 1
    assert run_context.logged_datasets[0]["context"] == "training"
    assert len(run_context.manifests) == 1
