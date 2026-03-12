"""Tests for the unified execute function with intelligent workflow routing."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from dlkit.interfaces.api import execute
from dlkit.interfaces.api.services.execution_service import (
    ExecutionService,
    WorkflowDetectionResult,
)
from dlkit.interfaces.api.domain import TrainingResult, OptimizationResult, WorkflowError
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.components.model_components import ModelComponentSettings
from dlkit.tools.config.optuna_settings import OptunaSettings
from dlkit.tools.config.session_settings import SessionSettings


class TestUnifiedExecuteFunction:
    """Test the unified execute function and its intelligent routing."""

    def test_execute_function_imports_correctly(self):
        """Test that the execute function can be imported."""
        from dlkit.interfaces.api import execute

        assert callable(execute)

    @patch("dlkit.interfaces.api.services.execution_service.TrainingService")
    def test_training_workflow_detection(self, mock_training_service):
        """Test that default settings route to training workflow."""
        # Setup
        mock_result = TrainingResult(
            model_state=Mock(), metrics={"loss": 0.5}, artifacts={}, duration_seconds=10.0
        )
        mock_training_service.return_value.execute_training.return_value = mock_result

        settings = GeneralSettings()

        # Execute
        result = execute(settings, epochs=10)

        # Verify
        assert isinstance(result, TrainingResult)
        mock_training_service.return_value.execute_training.assert_called_once()

    def test_inference_workflow_is_rejected(self):
        """Inference must use the dedicated predictor API, not execute()."""
        settings = GeneralSettings(
            SESSION=SessionSettings(inference=True),
            MODEL=ModelComponentSettings(name="LinearNetwork", checkpoint="model.ckpt"),
        )

        with pytest.raises(WorkflowError, match="load_model"):
            execute(settings)

    @patch("dlkit.interfaces.api.services.execution_service.OptimizationService")
    def test_optimization_workflow_detection(self, mock_optimization_service):
        """Test that Optuna settings route to optimization workflow."""
        # Setup
        mock_result = OptimizationResult(
            best_trial={"params": {"lr": 0.01}, "value": 0.95},
            training_result=Mock(),
            study_summary={"n_trials": 50},
            duration_seconds=300.0,
        )
        mock_optimization_service.return_value.execute_optimization.return_value = mock_result

        settings = GeneralSettings(
            OPTUNA=OptunaSettings(enabled=True, n_trials=50, direction="maximize")
        )

        # Execute
        result = execute(settings, trials=50)

        # Verify
        assert isinstance(result, OptimizationResult)
        mock_optimization_service.return_value.execute_optimization.assert_called_once()

    @patch.object(GeneralSettings, "__init__", return_value=None)
    def test_workflow_detection_priority_order(self, mock_init):
        """Optimization has priority over default training."""
        service = ExecutionService()

        optimization_settings = Mock()
        optimization_settings.SESSION = None
        optimization_settings.OPTUNA = Mock()
        optimization_settings.OPTUNA.enabled = True

        result = service._detect_workflow(optimization_settings)
        assert result.workflow_type == "optimization"
        assert result.optuna_enabled

        training_settings = Mock()
        training_settings.SESSION = None
        training_settings.OPTUNA = None

        result = service._detect_workflow(training_settings)
        assert result.workflow_type == "training"
        assert not result.optuna_enabled

    def test_inference_settings_raise_clear_error(self):
        """ExecutionService should reject inference settings explicitly."""
        service = ExecutionService()
        mock_settings = Mock()
        mock_settings.SESSION = Mock()
        mock_settings.SESSION.inference = True

        with pytest.raises(WorkflowError, match="load_model"):
            service.execute(mock_settings)

    @patch.object(ExecutionService, "_detect_workflow")
    def test_parameter_routing_by_workflow(self, mock_detect):
        """Test that parameters are correctly routed based on workflow type."""
        service = ExecutionService()

        # Mock the underlying services to verify parameter passing
        with (
            patch.object(service, "_execute_training") as mock_training,
            patch.object(service, "_execute_optimization") as mock_optimization,
        ):
            # Test training parameter routing
            mock_detect.return_value = WorkflowDetectionResult(
                workflow_type="training",
                service_class=Mock,
                reasoning="default workflow",
                optuna_enabled=False,
            )

            mock_settings = Mock()
            mock_settings.SESSION = None
            service.execute(mock_settings, epochs=20, batch_size=64)
            mock_training.assert_called_once()

            # Check positional arguments
            call_args = mock_training.call_args[0]
            # _execute_training(settings, checkpoint_path, root_dir, output_dir, data_dir, epochs, batch_size, learning_rate, experiment_name, run_name, **additional_overrides)
            assert call_args[5] == 20  # epochs
            assert call_args[6] == 64  # batch_size

            # Reset mocks
            mock_training.reset_mock()
            mock_optimization.reset_mock()

            # Test optimization parameter routing
            mock_detect.return_value = WorkflowDetectionResult(
                workflow_type="optimization",
                service_class=Mock,
                reasoning="settings.OPTUNA.enabled=True",
                optuna_enabled=True,
            )

            service.execute(mock_settings, trials=50, study_name="test_study")
            mock_optimization.assert_called_once()

            # Check positional arguments for optimization
            call_args = mock_optimization.call_args[0]
            # _execute_optimization(settings, trials, checkpoint_path, root_dir, output_dir, data_dir, study_name, experiment_name, run_name, **additional_overrides)
            assert call_args[1] == 50  # trials
            assert call_args[6] == "test_study"  # study_name


class TestExecutionServiceDetection:
    """Test the ExecutionService workflow detection logic."""

    def test_detect_workflow_reasoning(self):
        """Test that workflow detection provides clear reasoning."""
        service = ExecutionService()

        # Test optimization detection reasoning
        optuna_settings = Mock()
        optuna_settings.SESSION = None
        optuna_settings.OPTUNA = Mock()
        optuna_settings.OPTUNA.enabled = True

        result = service._detect_workflow(optuna_settings)
        assert result.reasoning == "settings.OPTUNA.enabled=True"

        # Test default training reasoning
        default_settings = Mock()
        default_settings.SESSION = None
        default_settings.OPTUNA = None

        result = service._detect_workflow(default_settings)
        assert "default workflow" in result.reasoning

    def test_inference_guard(self):
        """Inference settings should be rejected before workflow detection."""
        service = ExecutionService()
        inference_settings = Mock()
        inference_settings.SESSION = Mock()
        inference_settings.SESSION.inference = True

        with pytest.raises(WorkflowError, match="load_model"):
            service._ensure_not_inference(inference_settings)

    def test_optuna_detection_helper(self):
        """Test the Optuna detection helper method."""
        service = ExecutionService()

        # Test Optuna disabled
        disabled_settings = Mock()
        disabled_settings.OPTUNA = None
        assert not service._is_optuna_enabled(disabled_settings)

        # Test Optuna enabled
        enabled_settings = Mock()
        enabled_settings.OPTUNA = Mock()
        enabled_settings.OPTUNA.enabled = True
        assert service._is_optuna_enabled(enabled_settings)


class TestWorkflowDetectionResult:
    """Test the WorkflowDetectionResult dataclass."""

    def test_workflow_detection_result_creation(self):
        """Test WorkflowDetectionResult can be created with all fields."""
        result = WorkflowDetectionResult(
            workflow_type="optimization",
            service_class=ExecutionService,
            reasoning="test reasoning",
            optuna_enabled=True,
        )

        assert result.workflow_type == "optimization"
        assert result.service_class == ExecutionService
        assert result.reasoning == "test reasoning"
        assert result.optuna_enabled
