"""Integration tests for MLflow training workflows.

Tests the complete end-to-end pipeline from settings configuration
to MLflow strategy execution to final results logging.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname
from uuid import uuid4

import pytest

pytest.importorskip("mlflow")
import mlflow
from mlflow.tracking import MlflowClient

import dlkit
from dlkit import (
    get_model_version,
    list_model_versions,
    load_logged_model,
    load_registered_model,
    search_logged_models,
    search_registered_models,
)
from dlkit.core.models.wrappers.base import ProcessingLightningWrapper
from dlkit.tools.config import GeneralSettings


def _resolve_effective_model_name(model: Any) -> str:
    """Resolve tracked model class name, unwrapping DLKit Lightning wrappers."""
    if isinstance(model, ProcessingLightningWrapper):
        return type(model.model).__name__
    return type(model).__name__


def _expected_tracking_uri(result: Any | None = None) -> str:
    """Resolve tracking URI used by tests (result-first, then env, then current MLflow URI)."""
    if result is not None and getattr(result, "mlflow_tracking_uri", None):
        return result.mlflow_tracking_uri
    from_env = os.getenv("MLFLOW_TRACKING_URI")
    if from_env:
        return from_env
    return mlflow.get_tracking_uri()


@pytest.mark.slow
class TestMLflowTrainingIntegration:
    """Integration tests for MLflow behaviors not already covered by the sqlite smoke."""

    @pytest.mark.slow
    def test_mlflow_training_registers_model_and_supports_registry_lookup(
        self,
        mlflow_settings: GeneralSettings,
        tmp_path: Path,
    ) -> None:
        """Real E2E check: register model, resolve aliases, and load by name/version."""
        unique_suffix = uuid4().hex[:8]

        mlflow_cfg = mlflow_settings.MLFLOW
        mlflow_cfg_updated = mlflow_cfg.model_copy(
            update={
                "register_model": True,
                "experiment_name": f"registry_e2e_{unique_suffix}",
                "run_name": f"registry_run_{unique_suffix}",
                "registered_model_aliases": ("dataset_A_latest", "benchmark_high_precision"),
            },
        )
        settings_with_registration = mlflow_settings.model_copy(
            update={"MLFLOW": mlflow_cfg_updated}
        )

        result = dlkit.train(settings_with_registration)
        tracking_uri = _expected_tracking_uri(result)
        assert result.model_state is not None
        model_name = _resolve_effective_model_name(result.model_state.model)

        client = MlflowClient(tracking_uri=tracking_uri)
        versions = client.search_model_versions(f"name = '{model_name}'")
        assert versions, f"Expected registered versions for model '{model_name}'"
        latest_version = max(int(v.version) for v in versions)

        builtin_latest_version = client.get_model_version_by_alias(model_name, "latest")
        assert int(builtin_latest_version.version) == latest_version
        dataset_alias_version = client.get_model_version_by_alias(model_name, "dataset_A_latest")
        assert int(dataset_alias_version.version) == latest_version
        benchmark_alias_version = client.get_model_version_by_alias(
            model_name, "benchmark_high_precision"
        )
        assert int(benchmark_alias_version.version) == latest_version

        found_models = search_registered_models(model_name, tracking_uri=tracking_uri)
        assert found_models, f"Expected model '{model_name}' to be searchable"

        version_list = list_model_versions(model_name, tracking_uri=tracking_uri)
        assert latest_version in version_list

        version_entity = get_model_version(
            model_name,
            latest_version,
            tracking_uri=tracking_uri,
        )
        assert int(version_entity.version) == latest_version

        experiment = client.get_experiment_by_name(mlflow_cfg_updated.experiment_name)
        assert experiment is not None
        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        assert runs, "Expected a run in the registration experiment"
        run_info = runs[0].info

        artifact_uri = run_info.artifact_uri
        parsed_uri = urlparse(artifact_uri)
        artifact_path = Path(url2pathname(parsed_uri.path))
        assert artifact_path.exists(), f"Expected artifact path to exist: {artifact_path}"
        assert artifact_path.is_relative_to(tmp_path)

        root_artifacts = client.list_artifacts(run_info.run_id)
        artifact_paths = {a.path for a in root_artifacts}
        assert "lineage" in artifact_paths

        loaded_latest = load_registered_model(
            model_name,
            alias="latest",
            tracking_uri=tracking_uri,
        )
        loaded_by_version = load_registered_model(
            model_name,
            version=latest_version,
            tracking_uri=tracking_uri,
        )
        loaded_by_dataset_alias = load_registered_model(
            model_name,
            alias="dataset_A_latest",
            tracking_uri=tracking_uri,
        )
        loaded_by_benchmark_alias = load_registered_model(
            model_name,
            alias="benchmark_high_precision",
            tracking_uri=tracking_uri,
        )

        assert loaded_latest is not None
        assert loaded_by_version is not None
        assert loaded_by_dataset_alias is not None
        assert loaded_by_benchmark_alias is not None

    @pytest.mark.slow
    def test_mlflow_training_logs_model_without_registration_and_supports_logged_lookup(
        self,
        mlflow_settings: GeneralSettings,
        tmp_path: Path,
    ) -> None:
        """Real E2E check: unregistered runs still support model lookup/load via runs:/."""
        unique_suffix = uuid4().hex[:8]

        mlflow_cfg = mlflow_settings.MLFLOW
        mlflow_cfg_updated = mlflow_cfg.model_copy(
            update={
                "register_model": False,
                "experiment_name": f"logged_e2e_{unique_suffix}",
                "run_name": f"logged_run_{unique_suffix}",
            },
        )
        settings_without_registration = mlflow_settings.model_copy(
            update={"MLFLOW": mlflow_cfg_updated}
        )

        result = dlkit.train(settings_without_registration)
        tracking_uri = _expected_tracking_uri(result)
        assert result.model_state is not None
        model_name = _resolve_effective_model_name(result.model_state.model)

        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(mlflow_cfg_updated.experiment_name)
        assert experiment is not None
        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        assert runs, "Expected a run in the logged-model experiment"
        run_info = runs[0].info

        artifact_uri = run_info.artifact_uri
        parsed_uri = urlparse(artifact_uri)
        artifact_path = Path(url2pathname(parsed_uri.path))
        assert artifact_path.exists(), f"Expected artifact path to exist: {artifact_path}"
        assert artifact_path.is_relative_to(tmp_path)

        root_artifacts = client.list_artifacts(run_info.run_id)
        artifact_paths = {a.path for a in root_artifacts}
        assert "lineage" in artifact_paths

        logged_results = search_logged_models(
            model_name=model_name,
            experiment_name=mlflow_cfg_updated.experiment_name,
            tracking_uri=tracking_uri,
        )
        assert logged_results, "Expected logged model entries for non-registered run"
        latest_logged = logged_results[0]
        assert latest_logged.run_id == run_info.run_id

        loaded_logged = load_logged_model(
            model_uri=latest_logged.model_uri,
            tracking_uri=tracking_uri,
        )
        assert loaded_logged is not None
