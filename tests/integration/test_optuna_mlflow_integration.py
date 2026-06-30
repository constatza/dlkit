"""Integration tests for the distinct Optuna/MLflow seams worth keeping.

These tests keep one sqlite-backed combined workflow and one optuna-only smoke.
Server lifecycle and HTTP-backed behavior are intentionally not covered here.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest
from mlflow.tracking import MlflowClient

from dlkit.common import TrainingResult
from dlkit.engine.workflows.optimization.services import TrialExecutor
from dlkit.infrastructure.config.job_config import SearchJobConfig, TrainingJobConfig
from dlkit.interfaces.api import optimize as api_optimize

FAST_TEST_TIMEOUT = int(30 * float(os.getenv("DLKIT_TEST_TIMEOUT_MULTIPLIER", "1.0")))


def _artifact_path_from_uri(uri: str) -> Path:
    """Convert an artifact URI to a local filesystem path.

    Args:
        uri: Artifact URI string.

    Returns:
        Local Path derived from the URI.
    """
    parsed_uri = urlparse(uri)
    return Path(url2pathname(parsed_uri.path))


def _make_training_settings_dict() -> dict:
    """Build a minimal training section dict for SearchJobConfig.

    Returns:
        Dictionary suitable for the training section of SearchJobConfig.
    """
    return {
        "loss": "mse",
        "trainer": {
            "fast_dev_run": True,
            "enable_checkpointing": False,
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
        "optimizer": {"name": "AdamW", "lr": 1e-3},
        "metrics": [{"name": "MeanSquaredError", "module_path": "dlkit.domain.metrics"}],
    }


def _make_search_job_config(
    *,
    study_name: str,
    feature_path: Path,
    target_path: Path,
    storage: str | None = None,
    mlflow_uri: str | None = None,
    experiment_name: str = "optuna_mlflow_integration",
) -> SearchJobConfig:
    """Build a minimal SearchJobConfig for integration tests.

    Args:
        study_name: Optuna study name.
        feature_path: Path to feature .npy file.
        target_path: Path to target .npy file.
        storage: Optional Optuna storage URL.
        mlflow_uri: Optional MLflow tracking URI to enable MLflow tracking.
        experiment_name: MLflow experiment name.

    Returns:
        SearchJobConfig validated from dict.
    """
    payload: dict = {
        "run": {"type": "search", "seed": 42},
        "experiment": {"name": experiment_name},
        "model": {
            "class": "FFNN",
            "module_path": "dlkit.domain.nn",
            "hidden_size": 4,
            "num_layers": 0,
        },
        "data": {
            "class": "FlexibleDataset",
            "batch_size": 4,
            "num_workers": 0,
            "shuffle": True,
            "pin_memory": False,
            "persistent_workers": False,
            "features": [{"name": "x", "path": str(feature_path), "format": "npy"}],
            "targets": [{"name": "y", "path": str(target_path), "format": "npy"}],
        },
        "training": _make_training_settings_dict(),
        "search": {
            "n_trials": 1,
            "direction": "minimize",
            "study_name": study_name,
            "storage": storage,
            "space": {
                "model.hidden_size": {"type": "categorical", "choices": [2, 4]},
            },
        },
    }
    if mlflow_uri is not None:
        payload["tracking"] = {"backend": "mlflow", "uri": mlflow_uri}
    return SearchJobConfig.model_validate(payload)


@pytest.fixture
def stub_training_result() -> TrainingResult:
    """Fixture providing a minimal TrainingResult for stubbing trial execution.

    Returns:
        TrainingResult with stub metric values.
    """
    return TrainingResult(
        model_state=None,
        metrics={"loss": 0.125},
        artifacts={},
        duration_seconds=0.0,
    )


@pytest.fixture(autouse=True)
def stub_trial_execution(monkeypatch: pytest.MonkeyPatch, stub_training_result: TrainingResult):
    """Auto-use fixture that stubs out actual trial training to keep tests fast.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        stub_training_result: Stub result to return from each trial.
    """

    def _execute_trial(
        self,
        trial,
        base_settings,
        hyperparameters,
        trial_context=None,
        enable_checkpointing=False,
    ) -> TrainingResult:
        return stub_training_result

    monkeypatch.setattr(TrialExecutor, "execute_trial", _execute_trial)


@pytest.fixture
def combined_settings(
    minimal_dataset: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SearchJobConfig:
    """Create SearchJobConfig with both Optuna and MLflow enabled.

    Args:
        minimal_dataset: Fixture providing dataset paths.
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture for env var isolation.

    Returns:
        SearchJobConfig with Optuna and MLflow tracking configured.
    """
    import dlkit.engine.tracking.uri_resolver as uri_resolver

    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow_uri = f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_uri)
    monkeypatch.delenv("MLFLOW_ARTIFACT_URI", raising=False)
    monkeypatch.setattr(uri_resolver, "local_host_alive", lambda: False)

    experiment_name = f"test_optuna_mlflow_{tmp_path.name}"
    unique_storage = f"sqlite:///{(tmp_path / 'optuna.db').as_posix()}"

    return _make_search_job_config(
        study_name=f"test_study_{tmp_path.name}",
        feature_path=minimal_dataset["features"],
        target_path=minimal_dataset["targets"],
        storage=unique_storage,
        mlflow_uri=mlflow_uri,
        experiment_name=experiment_name,
    )


@pytest.fixture
def optuna_only_settings(
    minimal_dataset: dict[str, Path],
    tmp_path: Path,
) -> SearchJobConfig:
    """Create SearchJobConfig with only Optuna enabled (no MLflow).

    Args:
        minimal_dataset: Fixture providing dataset paths.
        tmp_path: Pytest temporary directory fixture.

    Returns:
        SearchJobConfig without MLflow tracking.
    """
    return _make_search_job_config(
        study_name=f"test_study_{tmp_path.name}",
        feature_path=minimal_dataset["features"],
        target_path=minimal_dataset["targets"],
        storage=f"sqlite:///{(tmp_path / 'optuna.db').as_posix()}",
    )


class TestOptunaMLflowOptimization:
    """Test that high-level optimize() API works with combined Optuna+MLflow settings."""

    @pytest.mark.timeout(FAST_TEST_TIMEOUT)
    def test_combined_settings_optimization(self, combined_settings: SearchJobConfig):
        """Combined Optuna+MLflow workflow should persist multiple MLflow runs."""
        import optuna

        result = api_optimize(combined_settings)

        assert result is not None
        assert hasattr(result, "duration_seconds")
        assert result.duration_seconds >= 0
        assert result.best_trial is not None
        storage_uri = combined_settings.search.storage
        assert isinstance(storage_uri, str)
        assert storage_uri.startswith("sqlite:///")
        storage_path = Path(storage_uri.removeprefix("sqlite:///"))
        assert storage_path.exists()

        client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
        assert combined_settings.experiment is not None
        experiment = client.get_experiment_by_name(combined_settings.experiment.name)
        assert experiment is not None

        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=10,
        )
        assert len(runs) >= 3

        run_names = {run.data.tags.get("mlflow.runName") for run in runs}
        assert f"best_retrain_trial_{result.best_trial.number}" in run_names
        assert f"trial_{result.best_trial.number}" in run_names

        for run in runs:
            artifact_path = _artifact_path_from_uri(run.info.artifact_uri)
            assert artifact_path.exists()
            assert artifact_path.is_relative_to(storage_path.parent)

        study = optuna.load_study(
            study_name=combined_settings.search.study_name,
            storage=storage_uri,
        )
        assert study.best_trial.number == result.best_trial.number
        assert study.best_trial.params == result.best_trial.params
        assert study.best_trial.value == pytest.approx(result.best_trial.value)

    @pytest.mark.timeout(FAST_TEST_TIMEOUT)
    def test_optuna_only_optimization(self, optuna_only_settings: SearchJobConfig):
        """Optuna-only optimization should succeed without MLflow persistence."""
        result = api_optimize(optuna_only_settings)

        assert result is not None
        assert hasattr(result, "duration_seconds")
        assert result.duration_seconds >= 0
        assert result.best_trial is not None

    def test_no_optimization_raises_error(self, minimal_dataset: dict[str, Path]) -> None:
        """Test that optimize() raises TypeError when passed a non-SearchJobConfig."""
        from dlkit.infrastructure.config.job_config import TrainingJobConfig

        training_cfg = TrainingJobConfig.model_validate(
            {
                "run": {"type": "train", "seed": 42},
                "experiment": {"name": "not_an_optimization"},
                "model": {
                    "class": "FFNN",
                    "module_path": "dlkit.domain.nn",
                    "hidden_size": 4,
                    "num_layers": 0,
                },
                "data": {
                    "class": "FlexibleDataset",
                    "batch_size": 4,
                    "num_workers": 0,
                    "features": [
                        {
                            "name": "x",
                            "path": str(minimal_dataset["features"]),
                            "format": "npy",
                        }
                    ],
                    "targets": [
                        {
                            "name": "y",
                            "path": str(minimal_dataset["targets"]),
                            "format": "npy",
                        }
                    ],
                },
                "training": _make_training_settings_dict(),
            }
        )

        with pytest.raises(TypeError):
            api_optimize(training_cfg)


class TestBackwardCompatibility:
    """Test that passing wrong config type raises a clear TypeError."""

    def test_vanilla_workflow_raises_error(self, training_settings: TrainingJobConfig):
        """Passing TrainingJobConfig to optimize() must raise TypeError."""
        with pytest.raises(TypeError, match="SearchJobConfig"):
            api_optimize(training_settings)


def test_null_object_pattern_through_apis(minimal_dataset: dict[str, Path]) -> None:
    """Optimization without MLflow should still work through the high-level API."""

    settings_no_mlflow = _make_search_job_config(
        study_name="test_study",
        feature_path=minimal_dataset["features"],
        target_path=minimal_dataset["targets"],
    )

    result = api_optimize(settings_no_mlflow)
    assert result is not None
    assert result.duration_seconds >= 0
