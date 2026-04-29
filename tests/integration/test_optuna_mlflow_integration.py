"""Integration tests for the distinct Optuna/MLflow seams worth keeping.

These tests keep one sqlite-backed combined workflow and one optuna-only smoke.
Server lifecycle and HTTP-backed behavior are intentionally not covered here.
"""

from __future__ import annotations

import os

import pytest
from mlflow.tracking import MlflowClient

from dlkit.infrastructure.config.mlflow_settings import MLflowSettings
from dlkit.infrastructure.config.optuna_settings import OptunaSettings
from dlkit.infrastructure.config.workflow_configs import OptimizationWorkflowConfig
from dlkit.interfaces.api import optimize as api_optimize

FAST_TEST_TIMEOUT = int(30 * float(os.getenv("DLKIT_TEST_TIMEOUT_MULTIPLIER", "1.0")))


@pytest.fixture
def combined_settings(training_settings: OptimizationWorkflowConfig, tmp_path, monkeypatch):
    """Create OptimizationWorkflowConfig with both Optuna and MLflow enabled."""
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}")
    monkeypatch.delenv("MLFLOW_ARTIFACT_URI", raising=False)

    experiment_name = f"test_optuna_mlflow_{tmp_path.name}"
    unique_storage = f"sqlite:///{(tmp_path / 'optuna.db').as_posix()}"

    return OptimizationWorkflowConfig(
        SESSION=training_settings.SESSION,
        TRAINING=training_settings.TRAINING,
        DATAMODULE=training_settings.DATAMODULE,
        DATASET=training_settings.DATASET,
        MODEL=training_settings.MODEL,
        MLFLOW=MLflowSettings(experiment_name=experiment_name),
        OPTUNA=OptunaSettings(
            enabled=True,
            n_trials=1,
            study_name=f"test_study_{tmp_path.name}",
            storage=unique_storage,
            model={"hidden_size": [2, 4]},
        ),
    )


@pytest.fixture
def optuna_only_settings(optuna_settings: OptimizationWorkflowConfig):
    """Create settings with only Optuna enabled."""
    return optuna_settings


class TestOptunaMLflowOptimization:
    """Test that high-level optimize() API works with combined Optuna+MLflow settings."""

    @pytest.mark.timeout(FAST_TEST_TIMEOUT)
    def test_combined_settings_optimization(self, combined_settings):
        """Combined Optuna+MLflow workflow should persist multiple MLflow runs."""
        result = api_optimize(combined_settings)

        assert result is not None
        assert hasattr(result, "duration_seconds")
        assert result.duration_seconds >= 0
        assert result.best_trial is not None

        client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
        experiment = client.get_experiment_by_name(combined_settings.MLFLOW.experiment_name)
        assert experiment is not None

        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=10,
        )
        assert len(runs) >= 2

    @pytest.mark.timeout(FAST_TEST_TIMEOUT)
    def test_optuna_only_optimization(self, optuna_only_settings):
        """Optuna-only optimization should succeed without MLflow persistence."""
        result = api_optimize(optuna_only_settings)

        assert result is not None
        assert hasattr(result, "duration_seconds")
        assert result.duration_seconds >= 0
        assert result.best_trial is not None

    def test_no_optimization_raises_error(self, training_settings):
        """Test that optimize() raises error when OPTUNA is not enabled."""
        from dlkit.common import WorkflowError

        settings = OptimizationWorkflowConfig(
            SESSION=training_settings.SESSION,
            TRAINING=training_settings.TRAINING,
            DATAMODULE=training_settings.DATAMODULE,
            DATASET=training_settings.DATASET,
            MODEL=training_settings.MODEL,
            OPTUNA=OptunaSettings(enabled=False),
        )

        with pytest.raises(WorkflowError) as exc_info:
            api_optimize(settings)

        assert "OPTUNA is not enabled" in str(exc_info.value)


class TestBackwardCompatibility:
    """Test that passing wrong config type raises a clear TypeError."""

    def test_vanilla_workflow_raises_error(self, training_settings):
        """Passing TrainingWorkflowConfig to optimize() must raise TypeError."""
        with pytest.raises(TypeError, match="OptimizationWorkflowConfig"):
            api_optimize(training_settings)


def test_null_object_pattern_through_apis(training_settings) -> None:
    """Optimization without MLflow should still work through the high-level API."""
    settings_no_mlflow = OptimizationWorkflowConfig(
        SESSION=training_settings.SESSION,
        TRAINING=training_settings.TRAINING,
        DATAMODULE=training_settings.DATAMODULE,
        DATASET=training_settings.DATASET,
        MODEL=training_settings.MODEL,
        OPTUNA=OptunaSettings(
            enabled=True,
            n_trials=1,
            direction="minimize",
            study_name="test_study",
            model={"hidden_size": [2, 4]},
        ),
    )

    result = api_optimize(settings_no_mlflow)
    assert result is not None
    assert result.duration_seconds >= 0
