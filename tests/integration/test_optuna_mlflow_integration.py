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
from dlkit.infrastructure.config import SessionSettings, TrainingSettings
from dlkit.infrastructure.config.mlflow_settings import MLflowSettings
from dlkit.infrastructure.config.model_components import MetricComponentSettings
from dlkit.infrastructure.config.optuna_settings import OptunaSettings
from dlkit.infrastructure.config.trainer_settings import TrainerSettings
from dlkit.infrastructure.config.workflow_configs import OptimizationWorkflowConfig
from dlkit.interfaces.api import optimize as api_optimize

FAST_TEST_TIMEOUT = int(30 * float(os.getenv("DLKIT_TEST_TIMEOUT_MULTIPLIER", "1.0")))


def _artifact_path_from_uri(uri: str) -> Path:
    parsed_uri = urlparse(uri)
    return Path(url2pathname(parsed_uri.path))


def _make_training_settings() -> TrainingSettings:
    return TrainingSettings(
        epochs=1,
        trainer=TrainerSettings.model_validate(
            {
                "fast_dev_run": True,
                "enable_checkpointing": False,
                "accelerator": "cpu",
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        ),
        metrics=(
            MetricComponentSettings(
                name="MeanSquaredError",
                module_path="dlkit.domain.metrics",
            ),
        ),
    )


def _make_optimization_settings(
    *,
    study_name: str,
    storage: str | None = None,
    mlflow_settings: MLflowSettings | None = None,
    optuna_enabled: bool = True,
) -> OptimizationWorkflowConfig:
    return OptimizationWorkflowConfig(
        SESSION=SessionSettings(name="optuna_mlflow_integration", workflow="optimize", seed=42),
        TRAINING=_make_training_settings(),
        MLFLOW=mlflow_settings,
        OPTUNA=OptunaSettings(
            enabled=optuna_enabled,
            n_trials=1,
            direction="minimize",
            study_name=study_name,
            storage=storage,
            model={"hidden_size": {"choices": [2, 4]}},
        ),
    )


@pytest.fixture
def stub_training_result() -> TrainingResult:
    return TrainingResult(
        model_state=None,
        metrics={"loss": 0.125},
        artifacts={},
        duration_seconds=0.0,
    )


@pytest.fixture(autouse=True)
def stub_trial_execution(monkeypatch: pytest.MonkeyPatch, stub_training_result: TrainingResult):
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
def combined_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create OptimizationWorkflowConfig with both Optuna and MLflow enabled."""
    import dlkit.engine.tracking.uri_resolver as uri_resolver

    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}")
    monkeypatch.delenv("MLFLOW_ARTIFACT_URI", raising=False)
    monkeypatch.setattr(uri_resolver, "local_host_alive", lambda: False)

    experiment_name = f"test_optuna_mlflow_{tmp_path.name}"
    unique_storage = f"sqlite:///{(tmp_path / 'optuna.db').as_posix()}"

    return _make_optimization_settings(
        study_name=f"test_study_{tmp_path.name}",
        storage=unique_storage,
        mlflow_settings=MLflowSettings(experiment_name=experiment_name),
    )


@pytest.fixture
def optuna_only_settings(tmp_path: Path):
    """Create settings with only Optuna enabled."""
    return _make_optimization_settings(
        study_name=f"test_study_{tmp_path.name}",
        storage=f"sqlite:///{(tmp_path / 'optuna.db').as_posix()}",
    )


class TestOptunaMLflowOptimization:
    """Test that high-level optimize() API works with combined Optuna+MLflow settings."""

    @pytest.mark.timeout(FAST_TEST_TIMEOUT)
    def test_combined_settings_optimization(self, combined_settings):
        """Combined Optuna+MLflow workflow should persist multiple MLflow runs."""
        import optuna

        result = api_optimize(combined_settings)

        assert result is not None
        assert hasattr(result, "duration_seconds")
        assert result.duration_seconds >= 0
        assert result.best_trial is not None
        storage_uri = combined_settings.OPTUNA.storage
        assert isinstance(storage_uri, str)
        assert storage_uri.startswith("sqlite:///")
        storage_path = Path(storage_uri.removeprefix("sqlite:///"))
        assert storage_path.exists()

        client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
        experiment = client.get_experiment_by_name(combined_settings.MLFLOW.experiment_name)
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
            study_name=combined_settings.OPTUNA.study_name,
            storage=storage_uri,
        )
        assert study.best_trial.number == result.best_trial.number
        assert study.best_trial.params == result.best_trial.params
        assert study.best_trial.value == pytest.approx(result.best_trial.value)

    @pytest.mark.timeout(FAST_TEST_TIMEOUT)
    def test_optuna_only_optimization(self, optuna_only_settings):
        """Optuna-only optimization should succeed without MLflow persistence."""
        result = api_optimize(optuna_only_settings)

        assert result is not None
        assert hasattr(result, "duration_seconds")
        assert result.duration_seconds >= 0
        assert result.best_trial is not None

    def test_no_optimization_raises_error(self):
        """Test that optimize() raises error when OPTUNA is not enabled."""
        from dlkit.common import WorkflowError

        settings = _make_optimization_settings(study_name="disabled_optuna", optuna_enabled=False)

        with pytest.raises(WorkflowError) as exc_info:
            api_optimize(settings)

        assert "OPTUNA is not enabled" in str(exc_info.value)


class TestBackwardCompatibility:
    """Test that passing wrong config type raises a clear TypeError."""

    def test_vanilla_workflow_raises_error(self, training_settings):
        """Passing TrainingWorkflowConfig to optimize() must raise TypeError."""
        with pytest.raises(TypeError, match="OptimizationWorkflowConfig"):
            api_optimize(training_settings)


def test_null_object_pattern_through_apis() -> None:
    """Optimization without MLflow should still work through the high-level API."""
    settings_no_mlflow = _make_optimization_settings(study_name="test_study")

    result = api_optimize(settings_no_mlflow)
    assert result is not None
    assert result.duration_seconds >= 0
