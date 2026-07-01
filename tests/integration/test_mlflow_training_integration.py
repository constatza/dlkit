"""Integration tests for MLflow training workflows.

Tests the complete end-to-end pipeline from settings configuration
to MLflow strategy execution to final results logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname
from uuid import uuid4

import pytest

pytest.importorskip("mlflow")
import mlflow
from mlflow.tracking import MlflowClient

from dlkit.engine.adapters.lightning.base import ProcessingLightningWrapper
from dlkit.engine.tracking.artifact_logger import (
    TAG_LOGGED_MODEL_ARTIFACT_PATH,
    TAG_LOGGED_MODEL_URI,
)
from dlkit.infrastructure.config.experiment_settings import ExperimentSettings
from dlkit.infrastructure.config.job_config import TrainingJobConfig
from dlkit.interfaces.api import train as api_train
from dlkit.mlflow import (
    get_model_version,
    has_checkpoint_artifact,
    list_model_versions,
    load_logged_model,
    load_registered_model,
    register_logged_model,
    search_logged_models,
    search_registered_models,
    set_registered_model_alias,
)


def _require_experiment_settings(settings: TrainingJobConfig) -> ExperimentSettings:
    """Extract and assert experiment settings are present.

    Args:
        settings: Training job configuration.

    Returns:
        ExperimentSettings from the job config.
    """
    experiment = settings.experiment
    assert experiment is not None
    return experiment


def _resolve_effective_model_name(model: Any) -> str:
    """Resolve tracked model class name, unwrapping DLKit Lightning wrappers.

    Args:
        model: Model or wrapper instance.

    Returns:
        Class name of the underlying model.
    """
    if isinstance(model, ProcessingLightningWrapper):
        return type(model.model).__name__
    return type(model).__name__


def _expected_tracking_uri(result: Any | None = None) -> str:
    """Resolve tracking URI used by tests (result-first, then current MLflow URI).

    Args:
        result: Optional training result with mlflow_tracking_uri attribute.

    Returns:
        Resolved tracking URI string.
    """
    if result is not None and getattr(result, "mlflow_tracking_uri", None):
        return result.mlflow_tracking_uri
    return mlflow.get_tracking_uri()


def _artifact_path_from_uri(uri: str) -> Path:
    """Convert an artifact URI to a local filesystem path.

    Args:
        uri: Artifact URI string.

    Returns:
        Local Path derived from the URI.
    """
    parsed_uri = urlparse(uri)
    return Path(url2pathname(parsed_uri.path))


def _list_artifact_paths(client: MlflowClient, run_id: str, artifact_path: str = "") -> set[str]:
    """Recursively list all artifact paths under a run.

    Args:
        client: MLflow tracking client.
        run_id: MLflow run identifier.
        artifact_path: Sub-path within the artifact store.

    Returns:
        Set of artifact path strings discovered recursively.
    """
    discovered: set[str] = set()
    for artifact in client.list_artifacts(run_id, path=artifact_path or None):
        discovered.add(artifact.path)
        if artifact.is_dir:
            discovered.update(_list_artifact_paths(client, run_id, artifact.path))
    return discovered


@pytest.mark.slow
class TestMLflowTrainingIntegration:
    """Integration tests for MLflow behaviors not already covered by the sqlite smoke."""

    @pytest.mark.slow
    def test_mlflow_training_registers_model_and_supports_registry_lookup(
        self,
        mlflow_settings: TrainingJobConfig,
        tmp_path: Path,
    ) -> None:
        """Real E2E check: explicitly register logged model and resolve aliases."""
        unique_suffix = uuid4().hex[:8]

        experiment_cfg = _require_experiment_settings(mlflow_settings)
        updated_experiment = experiment_cfg.model_copy(
            update={
                "name": f"registry_e2e_{unique_suffix}",
                "run_name": f"registry_run_{unique_suffix}",
            },
        )
        settings_with_registration = mlflow_settings.model_copy(
            update={"experiment": updated_experiment}
        )

        result = api_train(settings_with_registration)
        tracking_uri = _expected_tracking_uri(result)
        assert result.model_state is not None
        model_name = _resolve_effective_model_name(result.model_state.model)

        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(updated_experiment.name)
        assert experiment is not None
        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        assert runs, "Expected a run in the registration experiment"
        run_info = runs[0].info
        registered_version = register_logged_model(
            model_name,
            run_id=run_info.run_id,
            tracking_uri=tracking_uri,
        )
        latest_version = int(registered_version.version)
        set_registered_model_alias(
            model_name,
            alias="dataset_A_latest",
            version=latest_version,
            tracking_uri=tracking_uri,
        )
        set_registered_model_alias(
            model_name,
            alias="benchmark_high_precision",
            version=latest_version,
            tracking_uri=tracking_uri,
        )

        versions = client.search_model_versions(f"name = '{model_name}'")
        assert versions, f"Expected registered versions for model '{model_name}'"
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

        artifact_uri = run_info.artifact_uri
        artifact_path = _artifact_path_from_uri(artifact_uri)
        assert artifact_path.exists(), f"Expected artifact path to exist: {artifact_path}"
        assert artifact_path.is_relative_to(tmp_path)

        root_artifacts = client.list_artifacts(run_info.run_id)
        artifact_paths = {a.path for a in root_artifacts}
        assert "lineage" in artifact_paths

        loaded_latest = load_registered_model(
            model_name,
            version=latest_version,
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
        mlflow_settings: TrainingJobConfig,
        tmp_path: Path,
    ) -> None:
        """Real E2E check: unregistered runs still support model lookup/load via runs:/."""
        unique_suffix = uuid4().hex[:8]

        experiment_cfg = _require_experiment_settings(mlflow_settings)
        updated_experiment = experiment_cfg.model_copy(
            update={
                "name": f"logged_e2e_{unique_suffix}",
                "run_name": f"logged_run_{unique_suffix}",
            },
        )
        settings_without_registration = mlflow_settings.model_copy(
            update={"experiment": updated_experiment}
        )

        result = api_train(settings_without_registration)
        tracking_uri = _expected_tracking_uri(result)
        assert result.model_state is not None
        model_name = _resolve_effective_model_name(result.model_state.model)

        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(updated_experiment.name)
        assert experiment is not None
        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        assert runs, "Expected a run in the logged-model experiment"
        run_info = runs[0].info

        artifact_uri = run_info.artifact_uri
        artifact_path = _artifact_path_from_uri(artifact_uri)
        assert artifact_path.exists(), f"Expected artifact path to exist: {artifact_path}"
        assert artifact_path.is_relative_to(tmp_path)

        root_artifacts = client.list_artifacts(run_info.run_id)
        artifact_paths = {a.path for a in root_artifacts}
        assert "lineage" in artifact_paths

        logged_results = search_logged_models(
            model_name=model_name,
            experiment_name=updated_experiment.name,
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

    def test_mlflow_training_publishes_minimal_runtime_artifacts(
        self,
        mlflow_settings: TrainingJobConfig,
        tmp_path: Path,
    ) -> None:
        """Tracked training should publish only the contract-critical artifacts."""
        unique_suffix = uuid4().hex[:8]
        experiment_cfg = _require_experiment_settings(mlflow_settings)
        updated_experiment = experiment_cfg.model_copy(
            update={
                "name": f"artifact_e2e_{unique_suffix}",
                "run_name": f"artifact_run_{unique_suffix}",
            }
        )
        settings = mlflow_settings.model_copy(update={"experiment": updated_experiment})

        result = api_train(settings)
        tracking_uri = _expected_tracking_uri(result)
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(_require_experiment_settings(settings).name)
        assert experiment is not None

        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        assert runs, "Expected a tracked run for artifact verification"
        run_info = runs[0].info

        artifact_root = _artifact_path_from_uri(run_info.artifact_uri)
        assert artifact_root.exists()
        assert artifact_root.is_relative_to(tmp_path)

        artifact_paths = _list_artifact_paths(client, run_info.run_id)
        assert "lineage" in artifact_paths
        assert any(path.startswith("splits/") for path in artifact_paths)
        assert not has_checkpoint_artifact(run_info.run_id, tracking_uri=tracking_uri)
        assert run_info.run_id == result.mlflow_run_id
        assert runs[0].data.tags[TAG_LOGGED_MODEL_ARTIFACT_PATH] == "model"
        assert runs[0].data.tags[TAG_LOGGED_MODEL_URI].startswith("models:/")
