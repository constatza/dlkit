"""Tests for the thin unified execute API adapter."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dlkit.interfaces.api import execute
from dlkit.runtime.workflows.entrypoints.execution import execute as runtime_execute_impl
from dlkit.shared import OptimizationResult, TrainingResult, WorkflowError
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.model_components import ModelComponentSettings
from dlkit.tools.config.optuna_settings import OptunaSettings
from dlkit.tools.config.session_settings import SessionSettings


class TestUnifiedExecuteFunction:
    """Test API execution routing behavior."""

    def test_execute_function_imports_correctly(self) -> None:
        assert callable(execute)

    @patch("dlkit.interfaces.api.functions.execution.runtime_execute")
    def test_training_workflow_delegates_to_runtime(self, mock_runtime_execute) -> None:
        mock_runtime_execute.return_value = TrainingResult(
            model_state=None,
            metrics={"loss": 0.5},
            artifacts={},
            duration_seconds=10.0,
        )

        settings = GeneralSettings()
        result = execute(settings, overrides={"epochs": 10})

        assert isinstance(result, TrainingResult)
        mock_runtime_execute.assert_called_once()
        assert mock_runtime_execute.call_args.kwargs["overrides"] == {"epochs": 10}

    @patch("dlkit.interfaces.api.functions.execution.runtime_execute")
    def test_optimization_workflow_delegates_to_runtime(self, mock_runtime_execute) -> None:
        training_result = TrainingResult(
            model_state=None,
            metrics={},
            artifacts={},
            duration_seconds=1.0,
        )
        mock_runtime_execute.return_value = OptimizationResult(
            best_trial={"params": {"lr": 0.01}, "value": 0.95},
            training_result=training_result,
            study_summary={"n_trials": 50},
            duration_seconds=300.0,
        )

        settings = GeneralSettings(
            OPTUNA=OptunaSettings(enabled=True, n_trials=50, direction="maximize")
        )
        result = execute(settings, overrides={"trials": 50})

        assert isinstance(result, OptimizationResult)
        mock_runtime_execute.assert_called_once()
        assert mock_runtime_execute.call_args.kwargs["overrides"] == {"trials": 50}

    def test_runtime_execution_prioritizes_optimization(self) -> None:
        settings = GeneralSettings(
            OPTUNA=OptunaSettings(enabled=True, n_trials=7, direction="maximize")
        )
        training_result = TrainingResult(
            model_state=None,
            metrics={},
            artifacts={},
            duration_seconds=1.0,
        )

        with (
            patch("dlkit.runtime.workflows.entrypoints.execution.optimize") as mock_optimize,
            patch("dlkit.runtime.workflows.entrypoints.execution.train") as mock_train,
        ):
            mock_optimize.return_value = OptimizationResult(
                best_trial={"value": 1.0},
                training_result=training_result,
                study_summary={"n_trials": 7},
                duration_seconds=1.0,
            )
            runtime_execute_impl(settings, overrides={"trials": 7, "study_name": "test-study"})

        mock_optimize.assert_called_once()
        mock_train.assert_not_called()
        assert mock_optimize.call_args.args[1] == {"trials": 7, "study_name": "test-study"}

    def test_runtime_execution_rejects_inference_settings(self) -> None:
        settings = GeneralSettings(
            SESSION=SessionSettings(inference=True),
            MODEL=ModelComponentSettings(name="LinearNetwork", checkpoint="model.ckpt"),
        )

        with pytest.raises(WorkflowError, match="load_model"):
            runtime_execute_impl(settings)
