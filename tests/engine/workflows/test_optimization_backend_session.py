"""Tests for optimization backend-session lifecycle and reporting."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from dlkit.common.errors import WorkflowError
from dlkit.engine.workflows.optimization.infrastructure import (
    NullOptimizationBackendSession,
    OptunaOptimizationBackendSession,
    OptunaStudyRepository,
)
from dlkit.engine.workflows.optimization.value_objects import (
    OptimizationDirection,
    Study,
    Trial,
    TrialState,
)


class _FakeStorage:
    def __init__(self) -> None:
        self.remove_session_calls = 0

    def remove_session(self) -> None:
        self.remove_session_calls += 1


class _FakeOptunaTrial:
    def __init__(self, number: int = 0) -> None:
        self.number = number
        self.params: dict[str, object] = {}


class _FakeBackendStudy:
    def __init__(self) -> None:
        self._storage = _FakeStorage()
        self.ask_calls = 0
        self.tell_calls: list[tuple[_FakeOptunaTrial, object, object]] = []
        self.direction: object | None = None
        self.study_name = ""
        self.storage: str | None = None
        self.load_if_exists = False
        self.sampler: object | None = None
        self.pruner: object | None = None
        self.trials: list[object] = []
        self.best_trial: object | None = None

    def ask(self) -> _FakeOptunaTrial:
        trial = _FakeOptunaTrial(number=self.ask_calls)
        self.ask_calls += 1
        return trial

    def tell(self, trial: _FakeOptunaTrial, value: object, *, state: object) -> None:
        self.tell_calls.append((trial, value, state))


class _FakeOptunaModule:
    def __init__(self) -> None:
        self.study = SimpleNamespace(
            StudyDirection=SimpleNamespace(MINIMIZE="minimize", MAXIMIZE="maximize")
        )
        self.trial = SimpleNamespace(
            TrialState=SimpleNamespace(
                RUNNING="running",
                COMPLETE="complete",
                PRUNED="pruned",
                FAIL="fail",
            )
        )
        self.samplers = SimpleNamespace(TPESampler=lambda **_: object())
        self.pruners = SimpleNamespace(MedianPruner=lambda **_: object())
        self.created_studies: list[_FakeBackendStudy] = []

    def create_study(
        self,
        *,
        direction: object,
        sampler: object,
        pruner: object,
        study_name: str,
        storage: str | None,
        load_if_exists: bool,
    ) -> _FakeBackendStudy:
        backend_study = _FakeBackendStudy()
        backend_study.direction = direction
        backend_study.study_name = study_name
        backend_study.storage = storage
        backend_study.load_if_exists = load_if_exists
        backend_study.sampler = sampler
        backend_study.pruner = pruner
        backend_study.trials = []
        backend_study.best_trial = None
        self.created_studies.append(backend_study)
        return backend_study


def _make_study(repository: OptunaStudyRepository) -> Study:
    return repository.create_study(
        study_name="session-test",
        direction=OptimizationDirection.MINIMIZE,
        target_trials=1,
        sampler_config={"type": "TPESampler", "params": {}},
        pruner_config={"type": "MedianPruner", "params": {}},
        storage_config={"url": "sqlite:////tmp/session-test.db", "load_if_exists": True},
    )


def _make_trial(state: TrialState = TrialState.RUNNING, *, value: float | None = None) -> Trial:
    return Trial(
        trial_id="trial-0",
        trial_number=0,
        hyperparameters={},
        state=state,
        objective_value=value,
    )


def test_null_backend_session_is_noop() -> None:
    study = Study(
        study_id="study",
        study_name="study",
        direction=OptimizationDirection.MINIMIZE,
    )
    trial = _make_trial()

    with NullOptimizationBackendSession() as session:
        assert session.suggest_hyperparameters(study, trial, SimpleNamespace()) == {}
        assert session.report_trial_result(study, trial) is None


def test_optuna_backend_session_suggests_plain_hyperparameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optuna = _FakeOptunaModule()
    repository = OptunaStudyRepository(optuna_module=optuna)
    study = _make_study(repository)
    trial = _make_trial()

    class _FakeSettingsSampler:
        def sample(self, optuna_trial: _FakeOptunaTrial, base_settings: object) -> None:
            optuna_trial.params["hidden_size"] = 4

    monkeypatch.setattr(
        "dlkit.infrastructure.config.samplers.optuna_sampler.create_settings_sampler",
        lambda config: _FakeSettingsSampler(),
    )

    session = OptunaOptimizationBackendSession(repository.study_registry, optuna)
    with session:
        sampled = session.suggest_hyperparameters(
            study,
            trial,
            SimpleNamespace(OPTUNA=SimpleNamespace(model={"hidden_size": [2, 4]})),
        )

    assert sampled == {"hidden_size": 4}


@pytest.mark.parametrize(
    ("trial_state", "objective_value", "expected_backend_state", "expected_value"),
    [
        (TrialState.COMPLETE, 0.25, "complete", 0.25),
        (TrialState.PRUNED, None, "pruned", None),
        (TrialState.FAILED, None, "fail", None),
    ],
)
def test_optuna_backend_session_reports_terminal_trial_states(
    monkeypatch: pytest.MonkeyPatch,
    trial_state: TrialState,
    objective_value: float | None,
    expected_backend_state: str,
    expected_value: float | None,
) -> None:
    optuna = _FakeOptunaModule()
    repository = OptunaStudyRepository(optuna_module=optuna)
    study = _make_study(repository)
    trial = _make_trial()

    monkeypatch.setattr(
        "dlkit.infrastructure.config.samplers.optuna_sampler.create_settings_sampler",
        lambda config: SimpleNamespace(sample=lambda optuna_trial, base_settings: None),
    )

    session = OptunaOptimizationBackendSession(repository.study_registry, optuna)
    with session:
        session.suggest_hyperparameters(study, trial, SimpleNamespace(OPTUNA=SimpleNamespace()))
        completed_trial = replace(trial, state=trial_state, objective_value=objective_value)
        session.report_trial_result(study, completed_trial)

        backend_study = optuna.created_studies[0]
    assert len(backend_study.tell_calls) == 1
    reported_trial, reported_value, reported_state = backend_study.tell_calls[0]
    assert isinstance(reported_trial, _FakeOptunaTrial)
    assert reported_value == expected_value
    assert reported_state == expected_backend_state


def test_optuna_backend_session_rejects_duplicate_reporting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optuna = _FakeOptunaModule()
    repository = OptunaStudyRepository(optuna_module=optuna)
    study = _make_study(repository)
    trial = _make_trial()

    monkeypatch.setattr(
        "dlkit.infrastructure.config.samplers.optuna_sampler.create_settings_sampler",
        lambda config: SimpleNamespace(sample=lambda optuna_trial, base_settings: None),
    )

    session = OptunaOptimizationBackendSession(repository.study_registry, optuna)
    with session:
        session.suggest_hyperparameters(study, trial, SimpleNamespace(OPTUNA=SimpleNamespace()))
        completed_trial = replace(trial, state=TrialState.COMPLETE, objective_value=1.0)
        session.report_trial_result(study, completed_trial)

        with pytest.raises(WorkflowError, match="already been reported"):
            session.report_trial_result(study, completed_trial)


def test_optuna_backend_session_cleans_storage_and_mappings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optuna = _FakeOptunaModule()
    repository = OptunaStudyRepository(optuna_module=optuna)
    study = _make_study(repository)
    trial = _make_trial()

    monkeypatch.setattr(
        "dlkit.infrastructure.config.samplers.optuna_sampler.create_settings_sampler",
        lambda config: SimpleNamespace(sample=lambda optuna_trial, base_settings: None),
    )

    session = OptunaOptimizationBackendSession(repository.study_registry, optuna)
    with session:
        session.suggest_hyperparameters(study, trial, SimpleNamespace(OPTUNA=SimpleNamespace()))
        backend_study = optuna.created_studies[0]
        assert session._trial_mapping
        assert session._active_storages
        assert backend_study._storage.remove_session_calls == 0

    assert session._trial_mapping == {}
    assert session._reported_trials == set()
    assert session._active_storages == {}
    assert backend_study._storage.remove_session_calls == 1

    session.__exit__(None, None, None)
    assert backend_study._storage.remove_session_calls == 1


def test_optuna_backend_session_cleans_up_after_sampling_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optuna = _FakeOptunaModule()
    repository = OptunaStudyRepository(optuna_module=optuna)
    study = _make_study(repository)
    trial = _make_trial()

    def _raise_sampling_error(optuna_trial: _FakeOptunaTrial, base_settings: object) -> None:
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(
        "dlkit.infrastructure.config.samplers.optuna_sampler.create_settings_sampler",
        lambda config: SimpleNamespace(sample=_raise_sampling_error),
    )

    session = OptunaOptimizationBackendSession(repository.study_registry, optuna)

    with session, pytest.raises(RuntimeError, match="sampling failed"):
        session.suggest_hyperparameters(study, trial, SimpleNamespace(OPTUNA=SimpleNamespace()))

    backend_study = optuna.created_studies[0]
    assert session._trial_mapping == {}
    assert session._reported_trials == set()
    assert len(backend_study.tell_calls) == 1
    reported_trial, reported_value, reported_state = backend_study.tell_calls[0]
    assert isinstance(reported_trial, _FakeOptunaTrial)
    assert reported_value is None
    assert reported_state == "fail"
