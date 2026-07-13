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
    OptimizationDirection,
    Study,
    Trial,
    TrialFailedException,
    TrialPrunedException,
    TrialState,
)
from dlkit.infrastructure.config.job_config import SearchJobConfig
from dlkit.infrastructure.config.search_settings import (
    CategoricalParam,
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

    def extract_objective_value(
        self, training_result: TrainingResult, objective: str = "loss"
    ) -> float:
        # Ignores `objective` deliberately: this stand-in isn't wired to the
        # real val/loss vs train/loss convention, it just returns the fixed
        # "loss" key the fixtures set up.
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
        self.execute_best_retrain_calls: list[dict[str, Any]] = []
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

    def execute_best_retrain(
        self,
        base_settings: Any,
        hyperparameters: dict[str, Any],
        *,
        run_context: Any,
        tracker: Any,
        tracking_uri: str | None,
        hooks: Any,
    ) -> tuple[Any, TrainingResult]:
        """Recording stand-in for TrialExecutor.execute_best_retrain.

        Mirrors the real method's contract (applies hyperparameters, returns
        ``(trial_settings, result)``) without touching TrackingDecorator, so
        tests can assert this — not ``execute_trial`` — is the call the
        best-retrain leg makes.
        """
        trial_settings = self.apply_hyperparameters(base_settings, hyperparameters)
        self.execute_best_retrain_calls.append({"hyperparameters": dict(hyperparameters)})
        if self._exception is not None:
            raise self._exception
        return trial_settings, self._result

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


class _RecordingRunContext:
    """Plain duck-typed run context recording the generic ``IRunContext``-shaped
    calls the module-level logging functions in
    ``optimization.infrastructure.tracking`` (and ``fire_post_training_hooks``)
    actually make: ``log_params``, ``log_metrics``, ``log_artifact_content``,
    ``log_artifact``, plus ``is_active``/``tracking_uri`` for the best-retrain
    dispatch in ``OptimizationOrchestrator._retrain_best_trial``.

    ``ITrialRunContext``/``IStudyRunContext`` (which this used to subclass) no
    longer exist — production code now logs against the plain object a tracker
    yields, so this fake mirrors that plain shape instead.
    """

    def __init__(self, *, active: bool = True, tracking_uri: str | None = None) -> None:
        self.logged_params: dict[str, Any] = {}
        self.logged_metrics: dict[str, Any] = {}
        self.artifact_content_calls: list[tuple[Any, str]] = []
        self.artifact_calls: list[tuple[Any, str]] = []
        self._active = active
        self._tracking_uri = tracking_uri

    def log_params(self, params: dict[str, Any]) -> None:
        self.logged_params.update(params)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.logged_metrics.update(metrics)

    def log_artifact_content(self, content: Any, artifact_file: str) -> None:
        self.artifact_content_calls.append((content, artifact_file))

    def log_artifact(self, artifact_path: Any, artifact_dir: str = "") -> None:
        self.artifact_calls.append((artifact_path, artifact_dir))

    def set_tag(self, key: str, value: str) -> None:
        return None

    def is_active(self) -> bool:
        return self._active

    @property
    def tracking_uri(self) -> str | None:
        return self._tracking_uri


class _TrackingAdapter(IExperimentTracker):
    def __init__(self) -> None:
        self.study_context = _RecordingRunContext()
        self.trial_contexts: list[_RecordingRunContext] = []

    def __enter__(self) -> _TrackingAdapter:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def create_study_run(self, study: Study):
        from contextlib import nullcontext

        return nullcontext(self.study_context)

    def create_trial_run(self, trial: Trial, parent_context: Any):
        from contextlib import nullcontext

        context = _RecordingRunContext()
        self.trial_contexts.append(context)
        return nullcontext(context)

    def create_best_retrain_run(self, study: Study, parent_context: Any):
        from contextlib import nullcontext

        context = _RecordingRunContext()
        self.trial_contexts.append(context)
        return nullcontext(context)


@pytest.fixture
def search_settings_hidden_size() -> SearchSettings:
    return SearchSettings(
        space={"hidden_size": CategoricalParam(type="categorical", choices=[2, 4])}
    )


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
    # Ordinary trials and the best-retrain are no longer the same method with an
    # `enable_checkpointing` flag: ordinary trials always go through the
    # lightweight `execute_trial` path (checkpoints structurally disabled by
    # `execute_lightweight`), while the best retrain goes through the separate
    # `execute_best_retrain` method (full TrackingDecorator parity, checkpoints
    # included). With n_trials=1 that means exactly one call to each.
    assert len(trial_executor.execute_calls) == 1
    assert len(trial_executor.execute_best_retrain_calls) == 1
    assert len(trial_executor.apply_calls) == 2

    trial_context = tracker.trial_contexts[0]
    assert trial_context.artifact_content_calls[0][1] == "trial_config.toml"
    assert trial_context.logged_params["hidden_size"] == 2
    assert trial_context.logged_metrics["loss"] == 0.1

    # The outer study run also carries the best trial's training result, so it
    # looks like a regular training result at a glance, with no need to open
    # the nested best-retrain run.
    assert tracker.study_context.logged_metrics["loss"] == 0.1


def test_retrain_does_not_double_log_model_hyperparameters(
    minimal_training_result: TrainingResult,
) -> None:
    """The retrain run's model.* hyperparameters aren't re-logged via log_trial_hyperparameters.

    execute_best_retrain's full TrackingDecorator path already logs the
    model's resolved hparams (SettingsLogger.log_model_parameters) under the
    same stripped names log_trial_hyperparameters would use for "model."
    keys — logging them again here would duplicate the same MLflow param.
    """
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
        study_name="retrain-no-double-log",
        base_settings=_make_search_job(),
        n_trials=1,
        direction=OptimizationDirection.MINIMIZE,
    )

    assert result.best_trial is not None
    # trial_contexts[0] is the regular trial run, trial_contexts[1] is the retrain run.
    retrain_context = tracker.trial_contexts[1]
    assert "hidden_size" not in retrain_context.logged_params
    # Trial identifiers (unconditional in log_trial_hyperparameters) still get logged.
    assert retrain_context.logged_params["trial_id"] == result.best_trial.trial_id
    assert retrain_context.logged_params["trial_number"] == result.best_trial.trial_number


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
