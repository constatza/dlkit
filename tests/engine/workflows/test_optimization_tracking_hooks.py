"""Regression tests: LifecycleHooks.on_run_created fires for optimize()'s runs.

Closes the gap where search/optimize jobs silently dropped ``hooks`` (see
``engine/workflows/entrypoints/execution.py``'s ``SearchJobConfig`` branch),
leaving Optuna study/trial/best-retrain runs with no way to be linked to an
external parent run the way ``train()`` already supports.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import pytest

from dlkit.common.hooks import LifecycleHooks
from dlkit.engine.workflows.optimization.infrastructure.tracking import (
    MLflowTrackingAdapter,
)
from dlkit.engine.workflows.optimization.value_objects.models import (
    OptimizationDirection,
    Study,
    Trial,
    TrialState,
)


@dataclass
class _FakeRunContext:
    run_id: str


class _FakeTracker:
    """Minimal ``self._tracker.create_run`` double for the adapter under test."""

    def __init__(self, tracking_uri: str = "file:///tmp/mlruns") -> None:
        self._tracking_uri = tracking_uri
        self._run_ids = iter(["study-run", "trial-run", "retrain-run"])

    def get_tracking_uri(self) -> str:
        return self._tracking_uri

    @contextmanager
    def create_run(self, **_kwargs: object) -> Iterator[_FakeRunContext]:
        yield _FakeRunContext(run_id=next(self._run_ids))


@pytest.fixture
def recorded_run_creations() -> list[tuple[str, str | None]]:
    return []


@pytest.fixture
def hooks(recorded_run_creations: list[tuple[str, str | None]]) -> LifecycleHooks:
    def _record(run_id: str, tracking_uri: str | None) -> None:
        recorded_run_creations.append((run_id, tracking_uri))

    return LifecycleHooks(on_run_created=_record)


@pytest.fixture
def study_with_best_trial() -> Study:
    best = Trial(
        trial_id="trial-0",
        trial_number=0,
        hyperparameters={"lr": 0.1},
        objective_value=0.5,
        state=TrialState.COMPLETE,
    )
    return Study(
        study_id="study-1",
        study_name="study",
        direction=OptimizationDirection.MINIMIZE,
        trials=(best,),
    )


def test_create_study_run_fires_on_run_created(
    hooks: LifecycleHooks,
    recorded_run_creations: list[tuple[str, str | None]],
    study_with_best_trial: Study,
) -> None:
    adapter = MLflowTrackingAdapter(
        mlflow_tracker=_FakeTracker(), session_name="experiment", hooks=hooks
    )

    with adapter.create_study_run(study_with_best_trial):
        pass

    assert recorded_run_creations == [("study-run", "file:///tmp/mlruns")]


def test_create_trial_and_best_retrain_run_fire_on_run_created(
    hooks: LifecycleHooks,
    recorded_run_creations: list[tuple[str, str | None]],
    study_with_best_trial: Study,
) -> None:
    adapter = MLflowTrackingAdapter(
        mlflow_tracker=_FakeTracker(), session_name="experiment", hooks=hooks
    )
    trial = study_with_best_trial.trials[0]

    with adapter.create_study_run(study_with_best_trial) as study_context:
        with adapter.create_trial_run(trial, study_context):
            pass
        with adapter.create_best_retrain_run(study_with_best_trial, study_context):
            pass

    assert recorded_run_creations == [
        ("study-run", "file:///tmp/mlruns"),
        ("trial-run", "file:///tmp/mlruns"),
        ("retrain-run", "file:///tmp/mlruns"),
    ]


def test_no_hooks_means_no_op(study_with_best_trial: Study) -> None:
    adapter = MLflowTrackingAdapter(mlflow_tracker=_FakeTracker(), session_name="experiment")

    with adapter.create_study_run(study_with_best_trial):
        pass
