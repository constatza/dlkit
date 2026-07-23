"""Contract tests for the unified execute API adapter."""

from __future__ import annotations

from unittest.mock import Mock, patch

from dlkit.common import EvaluationResult, OptimizationResult, TrainingResult
from dlkit.engine.workflows.entrypoints.execution import execute as runtime_execute_impl
from dlkit.infrastructure.config.job_config import (
    InferenceJobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)
from dlkit.interfaces.api import execute
from dlkit.interfaces.api.domain.override_types import ExecutionOverrides


class TestUnifiedExecuteFunction:
    """Contract tests: given a config type, execute() produces the expected result type."""

    def test_execute_function_imports_correctly(self) -> None:
        assert callable(execute)

    @patch("dlkit.interfaces.api.functions.execution._executor.execute")
    def test_training_config_delegates_to_runtime(self, mock_executor_execute) -> None:
        mock_executor_execute.return_value = TrainingResult(
            model_state=None,
            metrics={"loss": 0.5},
            artifacts={},
            duration_seconds=10.0,
        )

        training_config = Mock(spec=TrainingJobConfig)

        result = execute(training_config, overrides=ExecutionOverrides(epochs=10))

        assert isinstance(result, TrainingResult)
        mock_executor_execute.assert_called_once()

    @patch("dlkit.interfaces.api.functions.execution._executor.execute")
    def test_optimization_config_delegates_to_runtime(self, mock_executor_execute) -> None:
        training_result = TrainingResult(
            model_state=None,
            metrics={},
            artifacts={},
            duration_seconds=1.0,
        )
        mock_executor_execute.return_value = OptimizationResult(
            best_trial=None,
            training_result=training_result,
            study_summary={"n_trials": 50},
            duration_seconds=300.0,
        )

        optimization_config = Mock(spec=SearchJobConfig)

        result = execute(optimization_config, overrides=ExecutionOverrides(trials=50))

        assert isinstance(result, OptimizationResult)
        mock_executor_execute.assert_called_once()

    @patch("dlkit.engine.workflows.entrypoints.execution.evaluate")
    def test_inference_config_delegates_to_evaluate(self, mock_evaluate) -> None:
        mock_evaluate.return_value = EvaluationResult(
            predictions=None,
            targets=None,
            metrics={"mae": 0.1},
            figures={},
            duration_seconds=1.0,
        )

        inference_config = Mock(spec=InferenceJobConfig)

        result = runtime_execute_impl(inference_config)

        assert isinstance(result, EvaluationResult)
        mock_evaluate.assert_called_once()
