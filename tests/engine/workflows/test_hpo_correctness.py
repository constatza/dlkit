"""Tests verifying the HPO bug in services.py is fixed.

Bug (services.py): sampled hyperparameters were never written back to the
Trial domain object, so TrialRecord.params was always {}.

These tests exercise the optimization service layer directly via small
mocks. No datasets, neural networks, or training infrastructure involved.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from dlkit.common import TrainingResult
from dlkit.engine.workflows.optimization.infrastructure import (
    OptunaOptimizationBackendSession,
    OptunaStudyRepository,
)
from dlkit.engine.workflows.optimization.services import (
    OptimizationOrchestrator,
    StudyManager,
)
from dlkit.engine.workflows.optimization.value_objects import (
    OptimizationDirection,
)
from dlkit.infrastructure.config.samplers.optuna_sampler import OptunaSettingsSampler
from dlkit.infrastructure.config.search_settings import CategoricalParam, SearchSettings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_training_result() -> TrainingResult:
    return TrainingResult(
        model_state=None,
        metrics={"loss": 0.1},
        artifacts={},
        duration_seconds=0.0,
    )


@pytest.fixture
def minimal_search_settings() -> SearchSettings:
    return SearchSettings(space={"x": CategoricalParam(type="categorical", choices=[2, 4])})


@pytest.fixture
def hidden_size_search_settings() -> SearchSettings:
    return SearchSettings(
        space={"hidden_size": CategoricalParam(type="categorical", choices=[2, 4])}
    )


@pytest.fixture
def minimal_sampler(minimal_search_settings: SearchSettings) -> OptunaSettingsSampler:
    return OptunaSettingsSampler(minimal_search_settings)


class _StubTrialExecutor:
    """Lightweight stand-in — returns a fixed TrainingResult without any ML."""

    def __init__(self, training_result: TrainingResult) -> None:
        self._result = training_result

    def execute_trial(
        self, trial, base_settings, hyperparameters, *args, **kwargs
    ) -> TrainingResult:
        return self._result

    def _extract_objective_value(self, training_result: TrainingResult) -> float:
        return training_result.metrics.get("loss", 0.0)

    def _apply_hyperparameters(self, base_settings: Any, hyperparameters: dict) -> Any:
        return base_settings


# ---------------------------------------------------------------------------
# Sampler: valid spec formats
# ---------------------------------------------------------------------------


class TestSamplerValidSpecs:
    """Only dict specs with 'low'/'high' or 'choices' are valid range specs."""

    def test_choices_dict_is_range_spec(self, minimal_sampler: OptunaSettingsSampler) -> None:
        assert minimal_sampler._is_range_specification({"choices": [2, 4]})

    def test_low_high_dict_is_range_spec(self, minimal_sampler: OptunaSettingsSampler) -> None:
        assert minimal_sampler._is_range_specification({"low": 1, "high": 10})

    def test_bare_list_is_not_range_spec(self, minimal_sampler: OptunaSettingsSampler) -> None:
        assert not minimal_sampler._is_range_specification([2, 4])

    def test_choices_list_of_lists_raises_validation_error(self) -> None:
        """Structured categorical choices should be rejected at config validation time."""
        with pytest.raises(ValidationError):
            CategoricalParam.model_validate({"type": "categorical", "choices": [[1, 2], [3, 4]]})

    def test_sampler_populates_trial_params_with_choices_spec(
        self, hidden_size_search_settings: SearchSettings
    ) -> None:
        import optuna

        sampler = OptunaSettingsSampler(hidden_size_search_settings)
        study = optuna.create_study(direction="minimize")
        optuna_trial = study.ask()
        sampler.sample(optuna_trial, SimpleNamespace(search=hidden_size_search_settings))

        assert "hidden_size" in optuna_trial.params
        assert optuna_trial.params["hidden_size"] in (2, 4)


# ---------------------------------------------------------------------------
# Bug: hyperparameters stored on Trial domain object
# ---------------------------------------------------------------------------


class TestHyperparametersStoredOnTrial:
    """OptimizationOrchestrator must persist sampled params on the Trial object."""

    def _make_orchestrator(
        self,
        training_result: TrainingResult,
    ) -> OptimizationOrchestrator:
        import optuna as _optuna

        # Both study_manager and backend_session must share the same OptunaStudyRepository
        # (and therefore the same study_registry) so domain studies are visible to the backend.
        repository = OptunaStudyRepository(optuna_module=_optuna)
        study_manager = StudyManager(repository)
        trial_executor = _StubTrialExecutor(training_result)
        backend_session = OptunaOptimizationBackendSession(repository.study_registry, _optuna)
        return OptimizationOrchestrator(
            study_manager=study_manager,
            trial_executor=trial_executor,
            optimization_backend_session=backend_session,
        )

    def test_three_trials_each_record_params(
        self,
        minimal_training_result: TrainingResult,
        hidden_size_search_settings: SearchSettings,
    ) -> None:
        orchestrator = self._make_orchestrator(minimal_training_result)
        base_settings = SimpleNamespace(search=hidden_size_search_settings)
        result = orchestrator.execute_optimization(
            study_name="hpo-correctness-3trials",
            base_settings=base_settings,
            n_trials=3,
            direction=OptimizationDirection.MINIMIZE,
        )

        assert result.successful_trials == 3, (
            f"Expected 3 successful trials; got {result.successful_trials}"
        )
        for trial in result.study.trials:
            assert "hidden_size" in trial.hyperparameters, (
                f"Trial {trial.trial_number} has empty hyperparameters: {trial.hyperparameters}"
            )
            assert trial.hyperparameters["hidden_size"] in (2, 4)

    def test_best_trial_hyperparameters_nonempty(
        self,
        minimal_training_result: TrainingResult,
        hidden_size_search_settings: SearchSettings,
    ) -> None:
        orchestrator = self._make_orchestrator(minimal_training_result)
        base_settings = SimpleNamespace(search=hidden_size_search_settings)
        result = orchestrator.execute_optimization(
            study_name="hpo-correctness-best",
            base_settings=base_settings,
            n_trials=2,
            direction=OptimizationDirection.MINIMIZE,
        )

        assert result.best_trial is not None
        assert result.best_trial.hyperparameters, (
            "best_trial.hyperparameters must not be empty — "
            "sampled params were not stored on the Trial domain object"
        )
