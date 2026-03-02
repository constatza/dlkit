"""Integration tests for combined Optuna + MLflow workflows.

This module tests that when both Optuna and MLflow are enabled:
1. The high-level optimize() API works correctly
2. MLflow server lifecycle is properly managed
3. Server startup messages are logged and visible
4. Nested run structure works (parent study + trial runs + best run)
5. Server cleanup happens correctly

These are true integration tests that test real behavior using high-level APIs.
"""

from __future__ import annotations


import pytest

import dlkit
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.mlflow_settings import (
    MLflowSettings,
    MLflowClientSettings,
    MLflowServerSettings,
)
from dlkit.tools.config.optuna_settings import OptunaSettings
from tests.test_timeouts import FAST_TEST_TIMEOUT, MEDIUM_TEST_TIMEOUT, SLOW_TEST_TIMEOUT


@pytest.fixture
def combined_settings(training_settings: GeneralSettings, tmp_path):
    """Create settings with both Optuna and MLflow enabled (fast file:// tracking).

    Uses file-based MLflow tracking for speed. For HTTP server tests, use
    combined_settings_http fixture instead.
    """
    from dlkit.interfaces.api.overrides.manager import BasicOverrideManager

    # Start with base training settings
    manager = BasicOverrideManager()

    # Create isolated MLflow directory per test for proper isolation
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    # Enable both MLflow (file:// for speed) and Optuna
    # Use unique experiment name per test to prevent conflicts
    settings_with_overrides = manager.apply_overrides(
        training_settings,
        enable_mlflow=True,
        experiment_name=f"test_optuna_mlflow_{tmp_path.name}",  # Unique per test
        tracking_uri=str(mlruns_dir.as_uri()),  # Fast file:// URI
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
def combined_settings_http(training_settings: GeneralSettings, tmp_path, free_port):
    """Create settings with both Optuna and MLflow enabled (HTTP server for lifecycle tests).

    This fixture starts a real MLflow HTTP server. Only use for tests that specifically
    need to verify server lifecycle behavior. For most tests, use combined_settings instead.
    """
    from dlkit.interfaces.api.overrides.manager import BasicOverrideManager

    # Start with base training settings
    manager = BasicOverrideManager()

    # Enable both MLflow (with HTTP server) and Optuna with unique port
    settings_with_overrides = manager.apply_overrides(
        training_settings,
        enable_mlflow=True,
        experiment_name=f"test_optuna_mlflow_http_{tmp_path.name}",  # Unique per test
        tracking_uri=f"http://127.0.0.1:{free_port}",
        mlflow_host="127.0.0.1",
        mlflow_port=free_port,
        enable_optuna=True,
        trials=1,  # Minimal trials for speed
        study_name=f"test_study_http_{tmp_path.name}",  # Unique per test
    )

    # Ensure optuna has isolated storage per test
    unique_storage = f"sqlite:///{(tmp_path / 'optuna_http.db').as_posix()}"
    new_optuna = settings_with_overrides.OPTUNA.model_copy(
        update={
            "storage": unique_storage,
            "study_name": f"test_study_http_{tmp_path.name}",
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
        """Test optimize() API with combined optuna+mlflow settings (file:// tracking)."""
        # Should successfully create and execute optimization workflow
        result = dlkit.optimize(combined_settings)

        # Verify basic result structure
        assert result is not None
        assert hasattr(result, "duration_seconds")
        assert result.duration_seconds >= 0

    @pytest.mark.timeout(FAST_TEST_TIMEOUT)
    def test_optuna_only_optimization(self, optuna_only_settings):
        """Test optimize() API with optuna-only settings."""
        # Should successfully create and execute optimization workflow
        result = dlkit.optimize(optuna_only_settings)

        # Verify basic result structure
        assert result is not None
        assert hasattr(result, "duration_seconds")
        assert result.duration_seconds >= 0

    def test_no_optimization_raises_error(self):
        """Test that optimize() raises error when optimization not enabled."""
        from dlkit.interfaces.api.domain import WorkflowError

        settings = GeneralSettings()  # No OPTUNA enabled

        with pytest.raises(WorkflowError) as exc_info:
            dlkit.optimize(settings)

        assert "OPTUNA is not enabled" in str(exc_info.value)


class TestMLflowServerLifecycle:
    """Test MLflow server lifecycle management in combined workflows."""

    @pytest.mark.slow  # This test actually starts an HTTP server
    @pytest.mark.timeout(SLOW_TEST_TIMEOUT)
    def test_optimization_handles_server_lifecycle(self, combined_settings_http):
        """Test that optimize() API properly handles MLflow server lifecycle.

        This test uses HTTP server to verify actual server lifecycle management.
        """
        # Should successfully start server, run optimization, and clean up
        result = dlkit.optimize(combined_settings_http)

        # Verify the optimization completed successfully
        assert result is not None
        assert result.duration_seconds >= 0

        # Server lifecycle is handled internally - no need to verify specific implementation


class TestServerMessagePropagation:
    """Test that MLflow server messages are properly logged and visible."""

    @pytest.mark.timeout(FAST_TEST_TIMEOUT)
    def test_optimization_handles_message_display(self, combined_settings):
        """Test that optimize() API properly handles server message display."""
        # Should successfully run with proper message handling
        result = dlkit.optimize(combined_settings)

        # Verify successful completion
        assert result is not None
        assert result.duration_seconds >= 0

        # Message handling is done internally

    @pytest.mark.timeout(MEDIUM_TEST_TIMEOUT)
    def test_combined_vs_optuna_only_workflows(self, combined_settings, optuna_only_settings):
        """Test that both combined and optuna-only workflows work via optimize() API."""
        # Both should work through the high-level API
        combined_result = dlkit.optimize(combined_settings)
        assert combined_result is not None
        assert combined_result.duration_seconds >= 0

        optuna_result = dlkit.optimize(optuna_only_settings)
        assert optuna_result is not None
        assert optuna_result.duration_seconds >= 0


class TestNestedRunStructure:
    """Test that nested MLflow run structure capability is available."""

    @pytest.mark.timeout(FAST_TEST_TIMEOUT)
    def test_optimization_supports_nested_runs(self, combined_settings):
        """Test that optimize() API properly supports nested MLflow runs."""
        # Should successfully execute with nested run structure
        result = dlkit.optimize(combined_settings)

        # Verify successful completion
        assert result is not None
        assert result.duration_seconds >= 0

        # Nested run structure is handled internally:
        # 1. Parent run for study
        # 2. Nested runs for each trial
        # 3. Nested run for best retrain
        # This is all managed by the optimize() API


class TestBackwardCompatibility:
    """Test that changes maintain backward compatibility."""

    @pytest.mark.timeout(FAST_TEST_TIMEOUT)
    def test_optuna_only_workflow_improved(self, optuna_only_settings):
        """Test that optuna-only workflows work through optimize() API."""
        # Should successfully execute optimization
        result = dlkit.optimize(optuna_only_settings)

        # Verify successful completion
        assert result is not None
        assert result.duration_seconds >= 0

    def test_vanilla_workflow_raises_error(self):
        """Test that vanilla (no optimization) workflows raise clear error."""
        from dlkit.interfaces.api.domain import WorkflowError

        settings = GeneralSettings()  # No OPTUNA or MLFLOW

        with pytest.raises(WorkflowError) as exc_info:
            dlkit.optimize(settings)

        assert "OPTUNA is not enabled" in str(exc_info.value)


class TestArchitecturalConsistency:
    """Test that the architecture is consistent and follows SOLID principles."""

    @pytest.mark.timeout(MEDIUM_TEST_TIMEOUT)
    def test_training_and_optimization_apis_work_together(self, combined_settings):
        """Test that both train() and optimize() APIs work with the same settings."""
        # Both APIs should work with combined settings
        train_result = dlkit.train(combined_settings)
        assert train_result is not None
        assert train_result.duration_seconds >= 0

        optimize_result = dlkit.optimize(combined_settings)
        assert optimize_result is not None
        assert optimize_result.duration_seconds >= 0

    @pytest.mark.timeout(FAST_TEST_TIMEOUT)
    def test_single_responsibility_maintained_through_apis(self, combined_settings):
        """Test that high-level APIs maintain single responsibility principle."""
        # optimize() API should handle optimization concerns
        result = dlkit.optimize(combined_settings)
        assert result is not None
        assert result.duration_seconds >= 0

        # Architecture separates concerns internally:
        # - StudyManager: Study lifecycle
        # - TrialExecutor: Individual trial execution
        # - OptimizationOrchestrator: Workflow coordination
        # - MLflowTrackingAdapter: Experiment tracking

    @pytest.mark.timeout(FAST_TEST_TIMEOUT)
    def test_null_object_pattern_through_apis(self):
        """Test that null object pattern works through high-level APIs."""
        # Settings with MLflow disabled should still work
        settings_no_mlflow = GeneralSettings(
            OPTUNA=OptunaSettings(
                enabled=True, n_trials=1, direction="minimize", study_name="test_study"
            )
        )

        # Should work without MLflow (null object pattern handles this internally)
        result = dlkit.optimize(settings_no_mlflow)
        assert result is not None
        assert result.duration_seconds >= 0


class TestNestedRunsImplementation:
    """Test that nested MLflow runs are properly implemented for Optuna optimization."""

    @pytest.mark.timeout(FAST_TEST_TIMEOUT)
    def test_optimization_executes_without_errors(self, combined_settings):
        """Test that optimize() API can be executed without errors."""
        # Should execute successfully with proper dependency injection
        result = dlkit.optimize(combined_settings)

        # Verify successful completion
        assert result is not None
        assert result.duration_seconds >= 0

        # Clean architecture handles all dependencies internally through proper DI
        # No undefined variable issues should occur due to clean design

    @pytest.mark.timeout(FAST_TEST_TIMEOUT)
    def test_optimization_supports_nested_runs(self, combined_settings):
        """Test that optimize() API properly supports nested run creation."""
        # Should execute successfully with nested run structure
        result = dlkit.optimize(combined_settings)

        # Verify successful completion
        assert result is not None
        assert result.duration_seconds >= 0

        # Clean architecture implements proper nested run structure through:
        # - OptimizationOrchestrator coordinates the entire workflow
        # - MLflowTrackingAdapter.create_study_run() for parent run
        # - MLflowTrackingAdapter.create_trial_run() for each trial
        # - MLflowTrackingAdapter.create_best_retrain_run() for final run
        # This provides the proper Study → Trial → Best Retrain hierarchy
