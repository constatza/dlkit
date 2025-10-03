"""Tests for the unified execute function with intelligent workflow routing."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from dlkit.interfaces.api import execute
from dlkit.interfaces.api.services.execution_service import (
    ExecutionService,
    WorkflowDetectionResult,
)
from dlkit.interfaces.api.domain import TrainingResult, InferenceResult, OptimizationResult
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.optuna_settings import OptunaSettings


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

    @patch("dlkit.interfaces.api.services.execution_service.InferenceService")
    @patch.object(ExecutionService, "_detect_workflow")
    def test_inference_workflow_detection(self, mock_detect, mock_inference_service):
        """Test that inference settings route to inference workflow."""
        # Setup
        mock_result = InferenceResult(
            model_state=Mock(), predictions=[1, 2, 3], metrics=None, duration_seconds=5.0
        )
        mock_inference_service.return_value.execute_inference.return_value = mock_result

        # Mock workflow detection to return inference
        mock_detect.return_value = WorkflowDetectionResult(
            workflow_type="inference",
            service_class=type(mock_inference_service.return_value),
            reasoning="settings.SESSION.inference=True",
            mlflow_enabled=False,
            optuna_enabled=False,
        )

        settings = GeneralSettings()

        # Execute
        result = execute(settings, checkpoint_path="model.ckpt")

        # Verify
        assert isinstance(result, InferenceResult)
        mock_inference_service.return_value.execute_inference.assert_called_once()

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
        """Test the priority order of workflow detection."""
        service = ExecutionService()

        # Create mock settings objects to avoid validation issues
        inference_settings = Mock()
        inference_settings.SESSION = Mock()
        inference_settings.SESSION.inference = True
        inference_settings.OPTUNA = Mock()
        inference_settings.OPTUNA.enabled = True
        inference_settings.MLFLOW = Mock()
        inference_settings.MLFLOW.enabled = True

        result = service._detect_workflow(inference_settings, "model.ckpt")
        assert result.workflow_type == "inference"

        # Test 2: Optimization has second priority
        optimization_settings = Mock()
        optimization_settings.SESSION = None
        optimization_settings.OPTUNA = Mock()
        optimization_settings.OPTUNA.enabled = True
        optimization_settings.MLFLOW = Mock()
        optimization_settings.MLFLOW.enabled = True

        result = service._detect_workflow(optimization_settings, None)
        assert result.workflow_type == "optimization"
        assert result.optuna_enabled
        assert result.mlflow_enabled

        # Test 3: Training is default
        training_settings = Mock()
        training_settings.SESSION = None
        training_settings.OPTUNA = None
        training_settings.MLFLOW = Mock()
        training_settings.MLFLOW.enabled = True

        result = service._detect_workflow(training_settings, None)
        assert result.workflow_type == "training"
        assert result.mlflow_enabled
        assert not result.optuna_enabled

    def test_inference_requires_checkpoint_path(self):
        """Test that inference workflow requires checkpoint_path."""
        service = ExecutionService()
        mock_settings = Mock()

        with pytest.raises(Exception) as exc_info:
            service._execute_inference(mock_settings, None, None, None, None, None)

        assert "checkpoint_path" in str(exc_info.value)

    @patch.object(ExecutionService, "_detect_workflow")
    def test_parameter_routing_by_workflow(self, mock_detect):
        """Test that parameters are correctly routed based on workflow type."""
        service = ExecutionService()

        # Mock the underlying services to verify parameter passing
        with (
            patch.object(service, "_execute_training") as mock_training,
            patch.object(service, "_execute_optimization") as mock_optimization,
            patch.object(service, "_execute_inference") as mock_inference,
        ):
            # Test training parameter routing
            mock_detect.return_value = WorkflowDetectionResult(
                workflow_type="training",
                service_class=Mock,
                reasoning="default workflow",
                mlflow_enabled=False,
                optuna_enabled=False,
            )

            mock_settings = Mock()
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
            mock_inference.reset_mock()

            # Test optimization parameter routing
            mock_detect.return_value = WorkflowDetectionResult(
                workflow_type="optimization",
                service_class=Mock,
                reasoning="settings.OPTUNA.enabled=True",
                mlflow_enabled=False,
                optuna_enabled=True,
            )

            service.execute(mock_settings, trials=50, study_name="test_study")
            mock_optimization.assert_called_once()

            # Check positional arguments for optimization
            call_args = mock_optimization.call_args[0]
            # _execute_optimization(settings, trials, checkpoint_path, root_dir, output_dir, data_dir, study_name, experiment_name, run_name, **additional_overrides)
            assert call_args[1] == 50  # trials
            assert call_args[6] == "test_study"  # study_name

            # Reset mocks
            mock_training.reset_mock()
            mock_optimization.reset_mock()
            mock_inference.reset_mock()

            # Test inference parameter routing
            mock_detect.return_value = WorkflowDetectionResult(
                workflow_type="inference",
                service_class=Mock,
                reasoning="settings.SESSION.inference=True",
                mlflow_enabled=False,
                optuna_enabled=False,
            )

            service.execute(mock_settings, checkpoint_path="model.ckpt", batch_size=32)
            mock_inference.assert_called_once()

            # Check positional arguments for inference
            call_args = mock_inference.call_args[0]
            # _execute_inference(settings, checkpoint_path, root_dir, output_dir, data_dir, batch_size, **additional_overrides)
            assert call_args[1] == "model.ckpt"  # checkpoint_path
            assert call_args[5] == 32  # batch_size


class TestExecutionServiceDetection:
    """Test the ExecutionService workflow detection logic."""

    def test_detect_workflow_reasoning(self):
        """Test that workflow detection provides clear reasoning."""
        service = ExecutionService()

        # Test inference detection reasoning
        inference_settings = Mock()
        inference_settings.SESSION = Mock()
        inference_settings.SESSION.inference = True

        result = service._detect_workflow(inference_settings, "model.ckpt")
        assert result.reasoning == "settings.SESSION.inference=True"

        # Test optimization detection reasoning
        optuna_settings = Mock()
        optuna_settings.SESSION = None
        optuna_settings.OPTUNA = Mock()
        optuna_settings.OPTUNA.enabled = True

        result = service._detect_workflow(optuna_settings, None)
        assert result.reasoning == "settings.OPTUNA.enabled=True"

        # Test default training reasoning
        default_settings = Mock()
        default_settings.SESSION = None
        default_settings.OPTUNA = None

        result = service._detect_workflow(default_settings, None)
        assert "default workflow" in result.reasoning

    def test_mlflow_detection_helper(self):
        """Test the MLflow detection helper method."""
        service = ExecutionService()

        # Test MLflow disabled
        disabled_settings = Mock()
        disabled_settings.MLFLOW = None
        assert not service._is_mlflow_enabled(disabled_settings)

        # Test MLflow enabled
        enabled_settings = Mock()
        enabled_settings.MLFLOW = Mock()
        enabled_settings.MLFLOW.enabled = True
        assert service._is_mlflow_enabled(enabled_settings)

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
            mlflow_enabled=True,
            optuna_enabled=True,
        )

        assert result.workflow_type == "optimization"
        assert result.service_class == ExecutionService
        assert result.reasoning == "test reasoning"
        assert result.mlflow_enabled
        assert result.optuna_enabled
