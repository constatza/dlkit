"""Integration tests for the distinct Optuna/MLflow seams worth keeping.

These tests keep one sqlite-backed combined workflow and one optuna-only smoke.
Server lifecycle and HTTP-backed behavior are intentionally not covered here.
"""

from __future__ import annotations

import os

import pytest
from mlflow.tracking import MlflowClient

import dlkit
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.optuna_settings import OptunaSettings
from tests._timeout_constants import FAST_TEST_TIMEOUT


@pytest.fixture
def combined_settings(training_settings: GeneralSettings, tmp_path, monkeypatch):
    """Create settings with both Optuna and MLflow enabled via isolated sqlite tracking.

    Uses sqlite tracking for speed and deterministic isolation.
    """
    from dlkit.interfaces.api.overrides.manager import BasicOverrideManager

    # Start with base training settings
    manager = BasicOverrideManager()

    # Create isolated MLflow directory per test for proper isolation
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}")
    monkeypatch.delenv("MLFLOW_ARTIFACT_URI", raising=False)

    # Enable both MLflow and Optuna with tiny sqlite-backed stores.
    # Use unique experiment name per test to prevent conflicts
    settings_with_overrides = manager.apply_overrides(
        training_settings,
        enable_mlflow=True,
        experiment_name=f"test_optuna_mlflow_{tmp_path.name}",  # Unique per test
        enable_optuna=True,
        trials=1,  # Minimal trials for speed
        study_name=f"test_study_{tmp_path.name}",  # Unique per test
    )

    # Ensure optuna has isolated storage per test
    unique_storage = f"sqlite:///{(tmp_path / 'optuna.db').as_posix()}"
    new_optuna = settings_with_overrides.OPTUNA.model_copy(
        update={
            "storage": unique_storage,
            "study_name": f"test_study_{tmp_path.name}",
        }
    )
    return settings_with_overrides.model_copy(update={"OPTUNA": new_optuna})


@pytest.fixture
def optuna_only_settings(optuna_settings: GeneralSettings):
    """Create settings with only Optuna enabled."""
    return optuna_settings


class TestOptunaMLflowOptimization:
    """Test that high-level optimize() API works with combined Optuna+MLflow settings."""

    @pytest.mark.timeout(FAST_TEST_TIMEOUT)
    def test_combined_settings_optimization(self, combined_settings):
        """Combined Optuna+MLflow workflow should persist multiple MLflow runs."""
        result = dlkit.optimize(combined_settings)

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
        result = dlkit.optimize(optuna_only_settings)

        assert result is not None
        assert hasattr(result, "duration_seconds")
        assert result.duration_seconds >= 0
        assert result.best_trial is not None

    def test_no_optimization_raises_error(self):
        """Test that optimize() raises error when optimization not enabled."""
        from dlkit.interfaces.api.domain import WorkflowError

        settings = GeneralSettings()  # No OPTUNA enabled

        with pytest.raises(WorkflowError) as exc_info:
            dlkit.optimize(settings)

        assert "OPTUNA is not enabled" in str(exc_info.value)


class TestBackwardCompatibility:
    """Test that changes maintain backward compatibility."""

    def test_vanilla_workflow_raises_error(self):
        """Test that vanilla (no optimization) workflows raise clear error."""
        from dlkit.interfaces.api.domain import WorkflowError

        settings = GeneralSettings()  # No OPTUNA or MLFLOW

        with pytest.raises(WorkflowError) as exc_info:
            dlkit.optimize(settings)

        assert "OPTUNA is not enabled" in str(exc_info.value)


def test_null_object_pattern_through_apis() -> None:
    """Optimization without MLflow should still work through the high-level API."""
    settings_no_mlflow = GeneralSettings(
        OPTUNA=OptunaSettings(
            enabled=True, n_trials=1, direction="minimize", study_name="test_study"
        )
    )

    result = dlkit.optimize(settings_no_mlflow)
    assert result is not None
    assert result.duration_seconds >= 0
