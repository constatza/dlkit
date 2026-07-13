"""Tests for TrialExecutor.extract_objective_value failing loudly instead of
silently returning 0.0 when the configured objective metric is missing or
non-numeric, and for OptimizationOrchestrator aborting the whole study (not
just one trial) when that happens.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from dlkit.common import TrainingResult, WorkflowError
from dlkit.engine.workflows.optimization.services import (
    OptimizationOrchestrator,
    TrialExecutor,
)
from dlkit.engine.workflows.optimization.value_objects import OptimizationDirection

from .test_hpo_correctness import _make_search_job, _make_study_manager, _RecordingBackendSession

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trial_executor() -> TrialExecutor:
    """Real TrialExecutor; build_factory is never invoked by extract_objective_value."""
    return TrialExecutor(build_factory=cast(Any, MagicMock()))


# ---------------------------------------------------------------------------
# TrialExecutor.extract_objective_value
# ---------------------------------------------------------------------------


class TestExtractObjectiveValue:
    def test_raises_when_objective_key_missing(self, trial_executor: TrialExecutor) -> None:
        result = TrainingResult(
            model_state=None, metrics={"train/loss": 0.4}, artifacts={}, duration_seconds=0.0
        )

        with pytest.raises(WorkflowError) as exc_info:
            trial_executor.extract_objective_value(result, objective="val/loss")

        assert exc_info.value.context["stage"] == "objective_extraction"
        assert exc_info.value.context["objective"] == "val/loss"

    def test_raises_when_objective_value_non_numeric(self, trial_executor: TrialExecutor) -> None:
        result = TrainingResult(
            model_state=None, metrics={"val/loss": [1, 2]}, artifacts={}, duration_seconds=0.0
        )

        with pytest.raises(WorkflowError) as exc_info:
            trial_executor.extract_objective_value(result, objective="val/loss")

        assert exc_info.value.context["stage"] == "objective_extraction"

    def test_returns_float_on_happy_path(self, trial_executor: TrialExecutor) -> None:
        result = TrainingResult(
            model_state=None, metrics={"val/loss": 0.25}, artifacts={}, duration_seconds=0.0
        )

        assert trial_executor.extract_objective_value(result, objective="val/loss") == 0.25


# ---------------------------------------------------------------------------
# OptimizationOrchestrator: fail-fast on a study-wide misconfiguration
# ---------------------------------------------------------------------------


class _MissingObjectiveTrialExecutor(TrialExecutor):
    """Real TrialExecutor whose execute_trial short-circuits training but keeps
    the real extract_objective_value, so the fail-fast path can be exercised
    end to end without any ML infrastructure."""

    def __init__(self, training_result: TrainingResult) -> None:
        super().__init__(build_factory=cast(Any, MagicMock()))
        self._result = training_result
        self.execute_trial_calls = 0

    def execute_trial(
        self, trial: Any, base_settings: Any, hyperparameters: dict, *args: Any, **kwargs: Any
    ) -> TrainingResult:
        self.execute_trial_calls += 1
        return self._result

    def apply_hyperparameters(self, base_settings: Any, hyperparameters: dict) -> Any:
        return base_settings


def test_missing_objective_aborts_whole_study_not_just_one_trial() -> None:
    """A config-typo'd search.objective must fail the whole optimize() call on
    the first trial, not silently degrade every trial to a bogus 0.0 and burn
    the full trial budget."""
    training_result = TrainingResult(
        model_state=None, metrics={"train/loss": 0.4}, artifacts={}, duration_seconds=0.0
    )
    trial_executor = _MissingObjectiveTrialExecutor(training_result)
    backend_session = _RecordingBackendSession(sampled={"model.hidden_size": 2})
    orchestrator = OptimizationOrchestrator(
        study_manager=_make_study_manager(),
        trial_executor=cast(TrialExecutor, trial_executor),
        optimization_backend_session=cast(Any, backend_session),
    )

    with pytest.raises(WorkflowError):
        orchestrator.execute_optimization(
            study_name="missing-objective",
            base_settings=_make_search_job(),
            n_trials=3,
            direction=OptimizationDirection.MINIMIZE,
        )

    assert trial_executor.execute_trial_calls == 1
