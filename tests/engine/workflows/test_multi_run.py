"""Tests for multi-run sweep orchestration.

Covers:
- RunVariant is a frozen dataclass (attribute assignment raises FrozenInstanceError)
- MultiRunOrchestrator satisfies IMultiRunOrchestrator protocol
- run_sweep() calls executor.execute exactly once per variant
- run_sweep() calls on_sweep_complete exactly once with (parent_run_ctx, results)
- run_sweep() returns a tuple of the correct length
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from dlkit.common.hooks import LifecycleHooks, RunCreatedEvent
from dlkit.common.results import TrainingResult
from dlkit.engine.training.components import RuntimeComponents
from dlkit.engine.workflows.multi_run import (
    IMultiRunOrchestrator,
    MultiRunOrchestrator,
    RunVariant,
)
from dlkit.infrastructure.config.experiment_settings import ExperimentSettings
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.run_settings import RunSettings
from dlkit.infrastructure.config.tracking_settings import TrackingSettings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_training_result() -> TrainingResult:
    """Minimal frozen TrainingResult used as executor return value.

    Returns:
        TrainingResult: Empty result with no model state or artifacts.
    """
    return TrainingResult(
        model_state=None,
        metrics={},
        artifacts={},
        duration_seconds=0.1,
    )


@pytest.fixture
def mock_executor(mock_training_result: TrainingResult) -> MagicMock:
    """Mock ITrainingExecutor whose execute() returns mock_training_result.

    Args:
        mock_training_result: Frozen result fixture.

    Returns:
        MagicMock: Executor mock.
    """
    executor = MagicMock()
    executor.execute = MagicMock(return_value=mock_training_result)
    return executor


@pytest.fixture
def job_config_settings() -> JobConfig:
    """Real JobConfig, required by TrackingDecorator.execute_within_run's isinstance check.

    Returns:
        JobConfig: Minimal but valid job config with mlflow tracking configured.
    """
    return JobConfig(
        run=RunSettings(type="train"),
        experiment=ExperimentSettings(name="test_experiment", run_name="test_run"),
        tracking=TrackingSettings(backend="mlflow"),
    )


@pytest.fixture
def runtime_components() -> RuntimeComponents:
    """Real RuntimeComponents, required since TrackingDecorator calls dataclasses.replace().

    Returns:
        RuntimeComponents: Components with a mock trainer/datamodule and a
        placeholder model. ``job_config_settings.model`` is None, so model
        hyperparameter logging no-ops without needing a real Lightning module.
    """

    @dataclass(frozen=True, slots=True)
    class DummyModel:
        pass

    trainer = MagicMock()
    trainer.callbacks = []

    return RuntimeComponents(
        model=cast("Any", DummyModel()),
        datamodule=MagicMock(),
        trainer=trainer,
        meta={},
    )


@pytest.fixture
def mock_build_factory(runtime_components: RuntimeComponents) -> MagicMock:
    """Mock BuildFactory whose build_components() returns a real RuntimeComponents.

    Args:
        runtime_components: Components fixture returned for every variant.

    Returns:
        MagicMock: Build factory mock.
    """
    build_factory = MagicMock()
    build_factory.build_components = MagicMock(return_value=runtime_components)
    return build_factory


@pytest.fixture
def mock_tracker() -> tuple[MagicMock, MagicMock]:
    """Mock MLflowTracker that acts as context manager at both levels.

    The tracker itself is a context manager (``__enter__`` returns itself).
    ``tracker.create_run()`` returns a context manager whose ``__enter__``
    yields a mock run context.

    Returns:
        Tuple of (tracker, run_ctx) where run_ctx is the value yielded by
        any create_run() context manager.
    """
    run_ctx = MagicMock()

    child_cm = MagicMock()
    child_cm.__enter__ = MagicMock(return_value=run_ctx)
    child_cm.__exit__ = MagicMock(return_value=False)

    tracker = MagicMock()
    tracker.__enter__ = MagicMock(return_value=tracker)
    tracker.__exit__ = MagicMock(return_value=False)
    tracker.create_run = MagicMock(return_value=child_cm)
    tracker.get_tracking_uri = MagicMock(return_value=None)
    tracker.is_local = MagicMock(return_value=False)

    return tracker, run_ctx


@pytest.fixture
def orchestrator(
    mock_build_factory: MagicMock,
    mock_executor: MagicMock,
    mock_tracker: tuple[MagicMock, MagicMock],
) -> MultiRunOrchestrator:
    """MultiRunOrchestrator wired with mock collaborators.

    Args:
        mock_build_factory: Build factory mock.
        mock_executor: Executor mock.
        mock_tracker: (tracker, run_ctx) tuple.

    Returns:
        MultiRunOrchestrator: Instance under test.
    """
    tracker, _ = mock_tracker
    return MultiRunOrchestrator(
        build_factory=mock_build_factory,
        executor=mock_executor,
        tracker=tracker,
    )


@pytest.fixture
def recorded_run_creations() -> list[RunCreatedEvent]:
    """Events recorded by the ``hooks`` fixture's ``on_run_created`` callable."""
    return []


@pytest.fixture
def hooks(recorded_run_creations: list[RunCreatedEvent]) -> LifecycleHooks:
    """LifecycleHooks recording every on_run_created firing."""
    return LifecycleHooks(on_run_created=recorded_run_creations.append)


@pytest.fixture
def orchestrator_with_hooks(
    mock_build_factory: MagicMock,
    mock_executor: MagicMock,
    mock_tracker: tuple[MagicMock, MagicMock],
    hooks: LifecycleHooks,
) -> MultiRunOrchestrator:
    """MultiRunOrchestrator wired with mock collaborators and recording hooks."""
    tracker, _ = mock_tracker
    return MultiRunOrchestrator(
        build_factory=mock_build_factory,
        executor=mock_executor,
        tracker=tracker,
        hooks=hooks,
    )


@pytest.fixture
def variant_a(job_config_settings: JobConfig) -> RunVariant:
    """First RunVariant for sweep tests.

    Args:
        job_config_settings: Real JobConfig shared across variants.

    Returns:
        RunVariant: Variant with run_name 'variant_a'.
    """
    return RunVariant(settings=job_config_settings, run_name="variant_a")


@pytest.fixture
def variant_b(job_config_settings: JobConfig) -> RunVariant:
    """Second RunVariant for sweep tests.

    Args:
        job_config_settings: Real JobConfig shared across variants.

    Returns:
        RunVariant: Variant with run_name 'variant_b'.
    """
    return RunVariant(settings=job_config_settings, run_name="variant_b")


# ---------------------------------------------------------------------------
# RunVariant tests
# ---------------------------------------------------------------------------


def test_run_variant_is_frozen(variant_a: RunVariant) -> None:
    """RunVariant is a frozen dataclass — attribute assignment must raise.

    Args:
        variant_a: A RunVariant fixture.
    """
    with pytest.raises(FrozenInstanceError):
        variant_a.run_name = "other"  # type: ignore[misc]


def test_run_variant_stores_run_name(variant_a: RunVariant) -> None:
    """RunVariant stores run_name correctly.

    Args:
        variant_a: RunVariant with run_name='variant_a'.
    """
    assert variant_a.run_name == "variant_a"


def test_run_variant_default_tags_is_empty_dict(job_config_settings: JobConfig) -> None:
    """RunVariant.tags defaults to an empty dict when not supplied.

    Args:
        job_config_settings: Real JobConfig fixture.
    """
    variant = RunVariant(settings=job_config_settings, run_name="x")
    assert variant.tags == {}


# ---------------------------------------------------------------------------
# Protocol satisfaction test
# ---------------------------------------------------------------------------


def test_multi_run_orchestrator_satisfies_protocol(
    orchestrator: MultiRunOrchestrator,
) -> None:
    """MultiRunOrchestrator is an instance of IMultiRunOrchestrator.

    Args:
        orchestrator: Fully wired orchestrator fixture.
    """
    assert isinstance(orchestrator, IMultiRunOrchestrator)


# ---------------------------------------------------------------------------
# run_sweep() tests
# ---------------------------------------------------------------------------


def test_run_sweep_calls_execute_once_per_variant(
    orchestrator: MultiRunOrchestrator,
    mock_executor: MagicMock,
    variant_a: RunVariant,
    variant_b: RunVariant,
) -> None:
    """run_sweep() calls executor.execute exactly once for each variant.

    Args:
        orchestrator: Orchestrator fixture.
        mock_executor: Executor mock to inspect call count.
        variant_a: First variant.
        variant_b: Second variant.
    """
    orchestrator.run_sweep(
        variants=[variant_a, variant_b],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )
    assert mock_executor.execute.call_count == 2


def test_run_sweep_returns_tuple_of_correct_length(
    orchestrator: MultiRunOrchestrator,
    variant_a: RunVariant,
    variant_b: RunVariant,
) -> None:
    """run_sweep() returns a tuple with one entry per variant.

    Args:
        orchestrator: Orchestrator fixture.
        variant_a: First variant.
        variant_b: Second variant.
    """
    results = orchestrator.run_sweep(
        variants=[variant_a, variant_b],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )
    assert isinstance(results, tuple)
    assert len(results) == 2


def test_run_sweep_calls_on_sweep_complete_once(
    orchestrator: MultiRunOrchestrator,
    mock_tracker: tuple[MagicMock, MagicMock],
    variant_a: RunVariant,
    variant_b: RunVariant,
) -> None:
    """run_sweep() calls on_sweep_complete exactly once before parent run closes.

    Args:
        orchestrator: Orchestrator fixture.
        mock_tracker: (tracker, run_ctx) tuple; run_ctx is the parent run context.
        variant_a: First variant.
        variant_b: Second variant.
    """
    callback = MagicMock()
    _, run_ctx = mock_tracker

    orchestrator.run_sweep(
        variants=[variant_a, variant_b],
        experiment_name="test_experiment",
        parent_run_name="parent",
        on_sweep_complete=callback,
    )

    callback.assert_called_once()


def test_run_sweep_on_sweep_complete_receives_parent_ctx_and_results(
    orchestrator: MultiRunOrchestrator,
    mock_tracker: tuple[MagicMock, MagicMock],
    mock_training_result: TrainingResult,
    variant_a: RunVariant,
    variant_b: RunVariant,
) -> None:
    """on_sweep_complete is called with (parent_run_ctx, results_tuple).

    Args:
        orchestrator: Orchestrator fixture.
        mock_tracker: (tracker, run_ctx) tuple.
        mock_training_result: Expected result returned per variant.
        variant_a: First variant.
        variant_b: Second variant.
    """
    callback = MagicMock()
    _, run_ctx = mock_tracker

    orchestrator.run_sweep(
        variants=[variant_a, variant_b],
        experiment_name="test_experiment",
        parent_run_name="parent",
        on_sweep_complete=callback,
    )

    called_args = callback.call_args[0]
    parent_ctx_arg, results_arg = called_args
    assert parent_ctx_arg is run_ctx
    assert isinstance(results_arg, tuple)
    assert len(results_arg) == 2


def test_run_sweep_without_callback_does_not_raise(
    orchestrator: MultiRunOrchestrator,
    variant_a: RunVariant,
) -> None:
    """run_sweep() with on_sweep_complete=None completes without error.

    Args:
        orchestrator: Orchestrator fixture.
        variant_a: Single variant sweep.
    """
    results = orchestrator.run_sweep(
        variants=[variant_a],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )
    assert len(results) == 1


def test_run_sweep_single_variant_execute_called_once(
    orchestrator: MultiRunOrchestrator,
    mock_executor: MagicMock,
    variant_a: RunVariant,
) -> None:
    """run_sweep() with a single variant calls executor.execute exactly once.

    Args:
        orchestrator: Orchestrator fixture.
        mock_executor: Executor mock.
        variant_a: Single variant.
    """
    orchestrator.run_sweep(
        variants=[variant_a],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )
    mock_executor.execute.assert_called_once()


def test_run_sweep_fires_on_run_created_for_parent_and_variants(
    orchestrator_with_hooks: MultiRunOrchestrator,
    recorded_run_creations: list[RunCreatedEvent],
    variant_a: RunVariant,
    variant_b: RunVariant,
) -> None:
    """run_sweep() fires on_run_created once for its own parent run
    (kind='sweep', is_outermost=True) and once per variant child run
    (kind='train', is_outermost=False)."""
    orchestrator_with_hooks.run_sweep(
        variants=[variant_a, variant_b],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )

    assert [event.kind for event in recorded_run_creations] == ["sweep", "train", "train"]
    assert [event.is_outermost for event in recorded_run_creations] == [True, False, False]
