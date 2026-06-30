"""Tests verifying the HPO bug in services.py is fixed.

Bug (services.py): sampled hyperparameters were never written back to the
Trial domain object, so TrialRecord.params was always {}.

These tests exercise the optimization service layer directly via small
mocks. No datasets, neural networks, or training infrastructure involved.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

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
    TrialExecutor,
)
from dlkit.engine.workflows.optimization.value_objects import (
    IExperimentTracker,
    IStudyRunContext,
    ITrialRunContext,
    OptimizationDirection,
    Study,
    Trial,
    TrialFailedException,
    TrialPrunedException,
    TrialState,
)
from dlkit.infrastructure.config.job_config import SearchJobConfig
from dlkit.infrastructure.config.samplers.interfaces import OptunaTrialProtocol
from dlkit.infrastructure.config.samplers.optuna_sampler import OptunaSettingsSampler
from dlkit.infrastructure.config.search_settings import (
    CategoricalParam,
    FloatParam,
    SearchSettings,
)

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


class _StubTrialExecutor:
    """Lightweight stand-in — returns a fixed TrainingResult without any ML."""

    def __init__(self, training_result: TrainingResult) -> None:
        self._result = training_result

    def execute_trial(
        self, trial, base_settings, hyperparameters, *args, **kwargs
    ) -> TrainingResult:
        return self._result

    def extract_objective_value(self, training_result: TrainingResult) -> float:
        return training_result.metrics.get("loss", 0.0)

    def apply_hyperparameters(self, base_settings: Any, hyperparameters: dict) -> Any:
        return base_settings


class _RecordingTrialExecutor(_StubTrialExecutor):
    def __init__(
        self,
        training_result: TrainingResult,
        *,
        exception: Exception | None = None,
    ) -> None:
        super().__init__(training_result)
        self._exception = exception
        self.execute_calls: list[dict[str, Any]] = []
        self.apply_calls: list[dict[str, Any]] = []

    def execute_trial(
        self,
        trial: Trial,
        base_settings: Any,
        hyperparameters: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> TrainingResult:
        self.execute_calls.append(
            {
                "trial_number": trial.trial_number,
                "hyperparameters": dict(hyperparameters),
                "kwargs": dict(kwargs),
            }
        )
        if self._exception is not None:
            raise self._exception
        return self._result

    def apply_hyperparameters(self, base_settings: Any, hyperparameters: dict) -> Any:
        self.apply_calls.append(dict(hyperparameters))
        return base_settings


class _RecordingBackendSession:
    def __init__(self, *, sampled: dict[str, Any]) -> None:
        self.sampled = sampled
        self.suggest_calls = 0
        self.report_calls = 0
        self.reported_states: list[TrialState] = []

    def __enter__(self) -> _RecordingBackendSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def suggest_hyperparameters(
        self, study: Study, trial: Trial, base_settings: SearchJobConfig
    ) -> dict[str, Any]:
        self.suggest_calls += 1
        return dict(self.sampled)

    def report_trial_result(self, study: Study, trial: Trial) -> None:
        self.report_calls += 1
        self.reported_states.append(trial.state)


class _RecordingTrialRunContext(ITrialRunContext):
    def __init__(self) -> None:
        self.pre_logs: list[dict[str, Any]] = []
        self.metric_logs: list[dict[str, Any]] = []
        self.artifact_logs: list[dict[str, Any]] = []

    def log_trial_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        self.pre_logs.append({"hyperparameters": dict(hyperparameters)})

    def log_trial_metrics(self, metrics: dict[str, Any]) -> None:
        self.metric_logs.append(dict(metrics))

    def log_trial_artifacts(self, artifacts: dict[str, Any]) -> None:
        self.artifact_logs.append(dict(artifacts))

    def log_trial_settings(self, settings: Any) -> None:
        self.pre_logs.append({"settings": settings})

    def log_model_hyperparameters(self, settings: Any) -> None:
        return None


class _StudyRunContext(IStudyRunContext):
    def log_study_metadata(self, study: Study) -> None:
        return None

    def log_study_summary(self, result) -> None:
        return None

    def log_best_trial_settings(self, settings: Any) -> None:
        return None


class _TrackingAdapter(IExperimentTracker):
    def __init__(self) -> None:
        self.study_context = _StudyRunContext()
        self.trial_contexts: list[_RecordingTrialRunContext] = []

    def __enter__(self) -> _TrackingAdapter:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def create_study_run(self, study: Study):
        from contextlib import nullcontext

        return nullcontext(self.study_context)

    def create_trial_run(self, trial: Trial, parent_context: IStudyRunContext):
        from contextlib import nullcontext

        context = _RecordingTrialRunContext()
        self.trial_contexts.append(context)
        return nullcontext(context)

    def create_best_retrain_run(self, study: Study, parent_context: IStudyRunContext):
        from contextlib import nullcontext

        context = _RecordingTrialRunContext()
        self.trial_contexts.append(context)
        return nullcontext(context)


# ---------------------------------------------------------------------------
# Sampler: valid spec formats
# ---------------------------------------------------------------------------


@pytest.fixture
def search_settings_choices() -> SearchSettings:
    return SearchSettings(space={"x": CategoricalParam(type="categorical", choices=[2, 4])})


@pytest.fixture
def search_settings_hidden_size() -> SearchSettings:
    return SearchSettings(
        space={"hidden_size": CategoricalParam(type="categorical", choices=[2, 4])}
    )


class TestSamplerValidSpecs:
    """Only dict specs with 'low'/'high' or 'choices' are valid range specs."""

    def test_choices_dict_is_range_spec(self, search_settings_choices: SearchSettings) -> None:
        sampler = OptunaSettingsSampler(search_settings_choices)
        assert sampler._is_range_specification({"choices": [2, 4]})

    def test_low_high_dict_is_range_spec(self) -> None:
        settings = SearchSettings(space={"x": FloatParam(type="float", low=1.0, high=10.0)})
        sampler = OptunaSettingsSampler(settings)
        assert sampler._is_range_specification({"low": 1, "high": 10})

    def test_bare_list_is_not_range_spec(self, search_settings_choices: SearchSettings) -> None:
        sampler = OptunaSettingsSampler(search_settings_choices)
        assert not sampler._is_range_specification([2, 4])

    def test_choices_list_of_lists_raises_validation_error(self) -> None:
        """Structured categorical choices should be rejected at config validation time."""
        with pytest.raises(ValidationError):
            SearchSettings(
                space={"layers": CategoricalParam(type="categorical", choices=[[1, 2], [2, 4]])}  # type: ignore  # intentional invalid input
            )

    def test_sampler_populates_trial_params_with_choices_spec(
        self, search_settings_hidden_size: SearchSettings
    ) -> None:
        import optuna

        sampler = OptunaSettingsSampler(search_settings_hidden_size)
        study = optuna.create_study(direction="minimize")
        optuna_trial = study.ask()
        job = cast(SearchJobConfig, SimpleNamespace(search=search_settings_hidden_size))
        sampler.sample(cast(OptunaTrialProtocol, optuna_trial), job)

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
            trial_executor=cast(TrialExecutor, trial_executor),
            optimization_backend_session=backend_session,
        )

    def test_three_trials_each_record_params(
        self,
        minimal_training_result: TrainingResult,
        search_settings_hidden_size: SearchSettings,
    ) -> None:
        orchestrator = self._make_orchestrator(minimal_training_result)

        base_settings = cast(SearchJobConfig, SimpleNamespace(search=search_settings_hidden_size))

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
        search_settings_hidden_size: SearchSettings,
    ) -> None:
        orchestrator = self._make_orchestrator(minimal_training_result)

        base_settings = cast(SearchJobConfig, SimpleNamespace(search=search_settings_hidden_size))

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


def _make_study_manager() -> StudyManager:
    repository = SimpleNamespace(
        create_study=lambda **kwargs: Study(
            study_id="study-1",
            study_name=kwargs["study_name"],
            direction=kwargs["direction"],
            target_trials=kwargs["target_trials"],
            sampler_config=kwargs.get("sampler_config"),
            pruner_config=kwargs.get("pruner_config"),
        ),
        get_study=lambda _study_id: None,
        save_study=lambda study: None,
    )
    return StudyManager(cast(Any, repository))


def _make_search_job() -> SearchJobConfig:
    return SearchJobConfig.model_validate(
        {
            "run": {"type": "search", "seed": 42},
            "experiment": {"name": "test-search"},
            "model": {"class": "DummyModel", "module_path": "dlkit.domain.nn"},
            "data": {
                "batch_size": 8,
                "num_workers": 0,
            },
            "training": {
                "loss": "mse",
                "trainer": {"max_epochs": 1, "accelerator": "cpu"},
                "optimizer": {"name": "AdamW", "lr": 1e-3},
            },
            "search": {
                "space": {
                    "model.hidden_size": {
                        "type": "categorical",
                        "choices": [2, 4],
                    }
                }
            },
        }
    )


def test_tracked_execution_samples_and_reports_once(
    minimal_training_result: TrainingResult,
) -> None:
    trial_executor = _RecordingTrialExecutor(minimal_training_result)
    backend_session = _RecordingBackendSession(sampled={"model.hidden_size": 2})
    tracker = _TrackingAdapter()
    orchestrator = OptimizationOrchestrator(
        study_manager=_make_study_manager(),
        trial_executor=cast(TrialExecutor, trial_executor),
        optimization_backend_session=cast(Any, backend_session),
        experiment_tracker=tracker,
    )

    result = orchestrator.execute_optimization(
        study_name="tracked-shared-path",
        base_settings=_make_search_job(),
        n_trials=1,
        direction=OptimizationDirection.MINIMIZE,
    )

    assert result.successful_trials == 1
    assert backend_session.suggest_calls == 1
    assert backend_session.report_calls == 1
    assert len(trial_executor.apply_calls) == 2
    assert len(trial_executor.execute_calls) == 2
    assert trial_executor.execute_calls[0]["kwargs"]["enable_checkpointing"] is False
    assert trial_executor.execute_calls[1]["kwargs"]["enable_checkpointing"] is True

    trial_context = tracker.trial_contexts[0]
    assert trial_context.pre_logs[0]["settings"] is not None
    assert trial_context.pre_logs[1]["hyperparameters"] == {"model.hidden_size": 2}
    assert trial_context.metric_logs == [{"loss": 0.1}]


@pytest.mark.parametrize(
    ("exception", "expected_state"),
    [
        (TrialPrunedException("pruned", pruned_at_step=3), TrialState.PRUNED),
        (TrialFailedException("failed"), TrialState.FAILED),
    ],
)
def test_tracked_and_untracked_paths_share_terminal_state_handling(
    minimal_training_result: TrainingResult,
    exception: Exception,
    expected_state: TrialState,
) -> None:
    base_settings = _make_search_job()

    untracked_executor = _RecordingTrialExecutor(minimal_training_result, exception=exception)
    untracked_backend = _RecordingBackendSession(sampled={"model.hidden_size": 2})
    untracked = OptimizationOrchestrator(
        study_manager=_make_study_manager(),
        trial_executor=cast(TrialExecutor, untracked_executor),
        optimization_backend_session=cast(Any, untracked_backend),
    )

    tracked_executor = _RecordingTrialExecutor(minimal_training_result, exception=exception)
    tracked_backend = _RecordingBackendSession(sampled={"model.hidden_size": 2})
    tracked = OptimizationOrchestrator(
        study_manager=_make_study_manager(),
        trial_executor=cast(TrialExecutor, tracked_executor),
        optimization_backend_session=cast(Any, tracked_backend),
        experiment_tracker=_TrackingAdapter(),
    )

    untracked_result = untracked.execute_optimization(
        study_name="untracked-terminal-state",
        base_settings=base_settings,
        n_trials=1,
        direction=OptimizationDirection.MINIMIZE,
    )
    tracked_result = tracked.execute_optimization(
        study_name="tracked-terminal-state",
        base_settings=base_settings,
        n_trials=1,
        direction=OptimizationDirection.MINIMIZE,
    )

    assert untracked_result.study.trials[0].state == expected_state
    assert tracked_result.study.trials[0].state == expected_state
    assert untracked_backend.report_calls == 1
    assert tracked_backend.report_calls == 1
