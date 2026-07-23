"""Shared fixtures for multi-run orchestration tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dlkit.common.hooks import LifecycleHooks, RunCreatedEvent
from dlkit.common.results import ChildFailure, TrainingResult
from dlkit.engine.workflows.multi_run import MultiRunOrchestrator, RunSpec
from dlkit.infrastructure.config.experiment_settings import ExperimentSettings
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.run_settings import RunSettings
from dlkit.infrastructure.config.tracking_settings import TrackingSettings

FIXTURES = Path(__file__).parent.parent.parent.parent / "fixtures" / "jobs"


@pytest.fixture
def simple_train_path() -> Path:
    """Path to the shared minimal single-file train TOML fixture.

    Returns:
        Absolute path to tests/fixtures/jobs/simple_train.toml.
    """
    return FIXTURES / "simple_train.toml"


@pytest.fixture
def mock_training_result() -> TrainingResult:
    """TrainingResult with a known mlflow_run_id, used as execute()'s return value.

    Returns:
        TrainingResult: Carries mlflow_run_id="child-run-id" so tests can
        assert post-hoc parent-tagging behavior.
    """
    return TrainingResult(
        model_state=None,
        metrics={},
        artifacts={},
        duration_seconds=0.1,
        mlflow_run_id="child-run-id",
    )


@pytest.fixture
def job_config_settings() -> JobConfig:
    """Real JobConfig with mlflow tracking configured.

    Returns:
        JobConfig: Minimal but valid job config.
    """
    return JobConfig(
        run=RunSettings(type="train"),
        experiment=ExperimentSettings(name="test_experiment", run_name="test_run"),
        tracking=TrackingSettings(backend="mlflow"),
    )


@pytest.fixture
def parent_run_ctx() -> MagicMock:
    """Mock IRunContext yielded by the mock tracker's parent create_run() call.

    Returns:
        MagicMock: Stubbed run context with a fixed run_id.
    """
    run_ctx = MagicMock()
    run_ctx.run_id = "parent-run-id"
    return run_ctx


@pytest.fixture
def mock_tracker(parent_run_ctx: MagicMock) -> MagicMock:
    """Mock MLflowTracker that acts as a context manager and yields parent_run_ctx.

    Args:
        parent_run_ctx: The run context yielded by create_run().

    Returns:
        MagicMock: Tracker mock.
    """
    parent_cm = MagicMock()
    parent_cm.__enter__ = MagicMock(return_value=parent_run_ctx)
    parent_cm.__exit__ = MagicMock(return_value=False)

    tracker = MagicMock()
    tracker.__enter__ = MagicMock(return_value=tracker)
    tracker.__exit__ = MagicMock(return_value=False)
    tracker.create_run = MagicMock(return_value=parent_cm)
    tracker.get_tracking_uri = MagicMock(return_value=None)
    tracker.is_local = MagicMock(return_value=False)
    tracker.set_run_tag = MagicMock()
    # Post-hoc handle for the (by then closed) parent run, used by
    # on_sweep_complete — same object as the one create_run() yielded, so
    # tests can assert identity against a single "the parent run" fixture.
    tracker.get_run_context = MagicMock(return_value=parent_run_ctx)
    return tracker


@pytest.fixture
def mock_execute(mock_training_result: TrainingResult) -> MagicMock:
    """Mock ChildDispatcher injected into MultiRunOrchestrator(dispatch=...).

    MultiRunOrchestrator takes its dispatcher (in production,
    entrypoints.execute) as an injected collaborator rather than importing it
    directly — engine.workflows may not depend on entrypoints (tach.toml) —
    so unit tests inject a mock here instead of monkeypatching an import.

    Args:
        mock_training_result: Default return value.

    Returns:
        MagicMock: Pre-configured to return mock_training_result; tests may
        override .return_value/.side_effect.
    """
    return MagicMock(return_value=mock_training_result)


@pytest.fixture
def orchestrator(mock_tracker: MagicMock, mock_execute: MagicMock) -> MultiRunOrchestrator:
    """MultiRunOrchestrator wired with the mock tracker/dispatcher, no hooks.

    Args:
        mock_tracker: Tracker mock fixture.
        mock_execute: Mock dispatcher fixture.

    Returns:
        MultiRunOrchestrator: Instance under test.
    """
    return MultiRunOrchestrator(tracker=mock_tracker, dispatch=mock_execute)


@pytest.fixture
def recorded_run_creations() -> list[RunCreatedEvent]:
    """Events recorded by the ``hooks`` fixture's ``on_run_created`` callable."""
    return []


@pytest.fixture
def recorded_child_failures() -> list[ChildFailure]:
    """Failures recorded by the ``hooks`` fixture's ``on_child_failed`` callable."""
    return []


@pytest.fixture
def hooks(
    recorded_run_creations: list[RunCreatedEvent],
    recorded_child_failures: list[ChildFailure],
) -> LifecycleHooks:
    """LifecycleHooks recording every on_run_created/on_child_failed firing."""
    return LifecycleHooks(
        on_run_created=recorded_run_creations.append,
        on_child_failed=recorded_child_failures.append,
    )


@pytest.fixture
def orchestrator_with_hooks(
    mock_tracker: MagicMock,
    mock_execute: MagicMock,
    hooks: LifecycleHooks,
) -> MultiRunOrchestrator:
    """MultiRunOrchestrator wired with the mock tracker/dispatcher and recording hooks."""
    return MultiRunOrchestrator(tracker=mock_tracker, dispatch=mock_execute, hooks=hooks)


@pytest.fixture
def spec_a(job_config_settings: JobConfig) -> RunSpec:
    """First RunSpec for sweep tests.

    Args:
        job_config_settings: Real JobConfig shared across specs.

    Returns:
        RunSpec: Spec with id/run_name "a"/"variant_a".
    """
    return RunSpec(id="a", label="a", settings=job_config_settings, run_name="variant_a")


@pytest.fixture
def spec_b(job_config_settings: JobConfig) -> RunSpec:
    """Second RunSpec for sweep tests.

    Args:
        job_config_settings: Real JobConfig shared across specs.

    Returns:
        RunSpec: Spec with id/run_name "b"/"variant_b".
    """
    return RunSpec(id="b", label="b", settings=job_config_settings, run_name="variant_b")
