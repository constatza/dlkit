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
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from dlkit.common import TrainingResult
from dlkit.common.hooks import LifecycleHooks, ParamValue
from dlkit.engine.tracking.lightweight_execution import fire_post_training_hooks
from dlkit.engine.tracking.tracking_decorator import TrackingDecorator
from dlkit.engine.training.components import RuntimeComponents
from dlkit.engine.workflows.optimization.infrastructure.tracking import (
    MLflowTrackingAdapter,
)
from dlkit.engine.workflows.optimization.value_objects.models import (
    OptimizationDirection,
    Study,
    Trial,
    TrialState,
)
from dlkit.infrastructure.config.experiment_settings import ExperimentSettings
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.run_settings import RunSettings
from dlkit.infrastructure.config.tracking_settings import TrackingSettings


class _FakeRunContext(MagicMock):
    """MagicMock-based run context: supports ``.run_id`` plus every generic
    ``IRunContext`` method (``log_params``/``set_tag``/``log_artifact``/etc.)
    that ``fire_post_training_hooks`` and ``TrackingDecorator`` call, so the
    same fake can drive both the ``on_run_created``-only tests and the new
    post-training-hooks tests without hand-rolling every method.
    """


class _FakeTracker:
    """Minimal ``self._tracker.create_run`` double for the adapter under test."""

    def __init__(self, tracking_uri: str = "file:///tmp/mlruns") -> None:
        self._tracking_uri = tracking_uri
        self._run_ids = iter(["study-run", "trial-run", "retrain-run"])

    def get_tracking_uri(self) -> str:
        return self._tracking_uri

    def is_local(self) -> bool:
        return False

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


# ---------------------------------------------------------------------------
# Regression: on_training_complete/extra_params/extra_tags/extra_artifacts now
# fire too — previously only on_run_created did. Ordinary trials fire these via
# `fire_post_training_hooks` (the lightweight per-trial path in
# `OptimizationOrchestrator._run_trial_iteration`); the best retrain fires them
# via `TrackingDecorator.execute_within_run`'s internal hook firing. Both are
# exercised here against the *real* run context `MLflowTrackingAdapter` yields.
# ---------------------------------------------------------------------------


@pytest.fixture
def post_training_calls() -> list[TrainingResult]:
    return []


@pytest.fixture
def training_result() -> TrainingResult:
    return TrainingResult(
        model_state=None, metrics={"loss": 0.2}, artifacts={}, duration_seconds=1.0
    )


@pytest.fixture
def extra_artifact_path(tmp_path: Path) -> Path:
    path = tmp_path / "extra_artifact.txt"
    path.write_text("extra artifact content")
    return path


@pytest.fixture
def full_hooks(
    recorded_run_creations: list[tuple[str, str | None]],
    post_training_calls: list[TrainingResult],
    extra_artifact_path: Path,
) -> LifecycleHooks:
    """LifecycleHooks with all 5 extension points wired to recording fixtures."""

    def _on_run_created(run_id: str, tracking_uri: str | None) -> None:
        recorded_run_creations.append((run_id, tracking_uri))

    def _on_training_complete(result: TrainingResult) -> None:
        post_training_calls.append(result)

    def _extra_params(result: TrainingResult) -> dict[str, ParamValue]:
        return {"final_loss": cast(float, result.metrics["loss"])}

    def _extra_tags(result: TrainingResult) -> dict[str, str]:
        return {"outcome": "complete"}

    def _extra_artifacts(result: TrainingResult) -> list[Path]:
        return [extra_artifact_path]

    return LifecycleHooks(
        on_run_created=_on_run_created,
        on_training_complete=_on_training_complete,
        extra_params=_extra_params,
        extra_tags=_extra_tags,
        extra_artifacts=_extra_artifacts,
    )


@pytest.fixture
def job_config_settings() -> JobConfig:
    """Real JobConfig; required by TrackingDecorator.execute_within_run's isinstance check."""
    return JobConfig(
        run=RunSettings(type="train"),
        experiment=ExperimentSettings(name="test_experiment", run_name="test_run"),
        tracking=TrackingSettings(backend="mlflow"),
    )


@pytest.fixture
def runtime_components() -> RuntimeComponents:
    """Real RuntimeComponents; model is a placeholder so hyperparameter logging no-ops."""

    @dataclass(frozen=True, slots=True)
    class _DummyModel:
        pass

    trainer = MagicMock()
    trainer.callbacks = []

    return RuntimeComponents(
        model=cast(Any, _DummyModel()),
        datamodule=MagicMock(),
        trainer=trainer,
        meta={},
    )


def test_fire_post_training_hooks_fires_all_four_for_trial_run(
    full_hooks: LifecycleHooks,
    post_training_calls: list[TrainingResult],
    training_result: TrainingResult,
    extra_artifact_path: Path,
    study_with_best_trial: Study,
) -> None:
    """The lightweight per-trial path fires on_training_complete/extra_params/
    extra_tags/extra_artifacts against the real run context
    MLflowTrackingAdapter.create_trial_run yields.
    """
    adapter = MLflowTrackingAdapter(
        mlflow_tracker=_FakeTracker(), session_name="experiment", hooks=full_hooks
    )
    trial = study_with_best_trial.trials[0]

    with adapter.create_study_run(study_with_best_trial) as study_context:
        with adapter.create_trial_run(trial, study_context) as trial_context:
            fire_post_training_hooks(full_hooks, trial_context, training_result)

    assert post_training_calls == [training_result]
    trial_context.log_params.assert_called_once_with({"final_loss": 0.2})
    trial_context.set_tag.assert_called_once_with("outcome", "complete")
    trial_context.log_artifact.assert_called_once_with(extra_artifact_path)


def test_execute_within_run_fires_all_four_hooks_for_best_retrain(
    full_hooks: LifecycleHooks,
    post_training_calls: list[TrainingResult],
    training_result: TrainingResult,
    extra_artifact_path: Path,
    study_with_best_trial: Study,
    job_config_settings: JobConfig,
    runtime_components: RuntimeComponents,
) -> None:
    """The best-retrain full-parity path (TrackingDecorator.execute_within_run)
    fires the same 4 post-training hooks against the real run context
    MLflowTrackingAdapter.create_best_retrain_run yields.
    """
    fake_tracker = _FakeTracker()
    adapter = MLflowTrackingAdapter(
        mlflow_tracker=fake_tracker, session_name="experiment", hooks=full_hooks
    )
    executor = MagicMock()
    executor.execute = MagicMock(return_value=training_result)

    with adapter.create_study_run(study_with_best_trial) as study_context:
        with adapter.create_best_retrain_run(
            study_with_best_trial, study_context
        ) as retrain_context:
            decorator = TrackingDecorator(executor, adapter.execution_tracker(), hooks=full_hooks)
            decorator.execute_within_run(
                runtime_components,
                job_config_settings,
                run_context=retrain_context,
                tracking_uri=fake_tracker.get_tracking_uri(),
            )

    assert post_training_calls == [training_result]
    retrain_context.log_params.assert_any_call({"final_loss": 0.2})
    retrain_context.set_tag.assert_any_call("outcome", "complete")
    retrain_context.log_artifact.assert_any_call(extra_artifact_path)
