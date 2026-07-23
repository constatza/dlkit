"""Tests for MultiRunOrchestrator.run_sweep().

Covers:
- MultiRunOrchestrator satisfies IMultiRunOrchestrator protocol
- run_sweep() calls entrypoints.execute() exactly once per child, with hooks forwarded
- run_sweep() patches each child's settings to the sweep's experiment/run_name
- run_sweep() returns a MultiRunResult with one ChildSuccess per child, in order
- run_sweep() calls on_sweep_complete exactly once with (parent_run_ctx, outcomes)
- run_sweep() without a callback does not raise
- run_sweep() fires on_run_created once for its own parent run (kind='sweep')
- run_sweep() tags a successful child's own MLflow run with the parent run id
- failure_policy="fail_fast" (default) propagates a child's exception immediately
- failure_policy="continue" records a ChildFailure and still runs later children
- failure_policy="continue_mark_parent_failed" additionally tags the parent run
"""

from __future__ import annotations

from unittest.mock import MagicMock

from dlkit.common.hooks import LifecycleHooks, RunCreatedEvent
from dlkit.common.results import ChildFailure, ChildSuccess, MultiRunResult, TrainingResult
from dlkit.engine.workflows.multi_run import IMultiRunOrchestrator, MultiRunOrchestrator, RunSpec

# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def test_multi_run_orchestrator_satisfies_protocol(orchestrator: MultiRunOrchestrator) -> None:
    """MultiRunOrchestrator is an instance of IMultiRunOrchestrator.

    Args:
        orchestrator: Fully wired orchestrator fixture.
    """
    assert isinstance(orchestrator, IMultiRunOrchestrator)


# ---------------------------------------------------------------------------
# Dispatch and result shape
# ---------------------------------------------------------------------------


def test_run_sweep_calls_execute_once_per_child(
    orchestrator: MultiRunOrchestrator,
    mock_execute: MagicMock,
    spec_a: RunSpec,
    spec_b: RunSpec,
) -> None:
    """run_sweep() calls entrypoints.execute() exactly once for each child.

    Args:
        orchestrator: Orchestrator fixture.
        mock_execute: Patched execute() to inspect call count.
        spec_a: First child spec.
        spec_b: Second child spec.
    """
    orchestrator.run_sweep(
        children=[spec_a, spec_b],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )
    assert mock_execute.call_count == 2


def test_run_sweep_forwards_hooks_to_execute(
    orchestrator_with_hooks: MultiRunOrchestrator,
    mock_execute: MagicMock,
    hooks: LifecycleHooks,
    spec_a: RunSpec,
) -> None:
    """run_sweep() forwards its own hooks into each execute() call.

    Args:
        orchestrator_with_hooks: Orchestrator fixture wired with hooks.
        mock_execute: Patched execute() to inspect call kwargs.
        hooks: The same LifecycleHooks instance orchestrator_with_hooks was built with.
        spec_a: Single child spec.
    """
    orchestrator_with_hooks.run_sweep(
        children=[spec_a],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )
    assert mock_execute.call_args.kwargs["hooks"] is hooks


def test_run_one_patches_experiment_and_run_name(
    orchestrator: MultiRunOrchestrator,
    mock_execute: MagicMock,
    spec_a: RunSpec,
) -> None:
    """_run_one() patches settings.experiment to the sweep's experiment/run_name.

    Regression guard for children landing in the wrong MLflow experiment,
    which would break post-hoc discoverability via find_child_run_ids().

    Args:
        orchestrator: Orchestrator fixture.
        mock_execute: Patched execute() to inspect the settings it was called with.
        spec_a: Single child spec, run_name="variant_a".
    """
    orchestrator.run_sweep(
        children=[spec_a],
        experiment_name="my-sweep-experiment",
        parent_run_name="parent",
    )
    called_settings = mock_execute.call_args.args[0]
    assert called_settings.experiment.name == "my-sweep-experiment"
    assert called_settings.experiment.run_name == spec_a.run_name


def test_run_sweep_returns_multi_run_result_with_child_successes(
    orchestrator: MultiRunOrchestrator,
    spec_a: RunSpec,
    spec_b: RunSpec,
    mock_training_result: TrainingResult,
) -> None:
    """run_sweep() returns a MultiRunResult with one ChildSuccess per child.

    Args:
        orchestrator: Orchestrator fixture.
        spec_a: First child spec.
        spec_b: Second child spec.
        mock_training_result: Expected wrapped result.
    """
    result = orchestrator.run_sweep(
        children=[spec_a, spec_b],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )
    assert isinstance(result, MultiRunResult)
    assert len(result.children) == 2
    for outcome, spec in zip(result.children, [spec_a, spec_b], strict=True):
        assert isinstance(outcome, ChildSuccess)
        assert outcome.child_id == spec.id
        assert outcome.result is mock_training_result


def test_run_sweep_tags_child_run_with_parent_run_id(
    orchestrator: MultiRunOrchestrator,
    mock_tracker: MagicMock,
    spec_a: RunSpec,
    mock_training_result: TrainingResult,
) -> None:
    """A successful child's own MLflow run gets tagged with the parent run id.

    Args:
        orchestrator: Orchestrator fixture.
        mock_tracker: Tracker mock exposing set_run_tag for inspection.
        spec_a: Single child spec.
        mock_training_result: Carries mlflow_run_id="child-run-id".
    """
    orchestrator.run_sweep(
        children=[spec_a],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )
    mock_tracker.set_run_tag.assert_called_once_with(
        mock_training_result.mlflow_run_id, "mlflow.parentRunId", "parent-run-id"
    )


def test_run_sweep_calls_on_sweep_complete_once(
    orchestrator: MultiRunOrchestrator,
    parent_run_ctx: MagicMock,
    spec_a: RunSpec,
    spec_b: RunSpec,
) -> None:
    """run_sweep() calls on_sweep_complete exactly once before parent run closes.

    Args:
        orchestrator: Orchestrator fixture.
        parent_run_ctx: The mock parent run context.
        spec_a: First child spec.
        spec_b: Second child spec.
    """
    callback = MagicMock()

    orchestrator.run_sweep(
        children=[spec_a, spec_b],
        experiment_name="test_experiment",
        parent_run_name="parent",
        on_sweep_complete=callback,
    )

    callback.assert_called_once()
    called_args = callback.call_args[0]
    parent_ctx_arg, outcomes_arg = called_args
    assert parent_ctx_arg is parent_run_ctx
    assert len(outcomes_arg) == 2


def test_run_sweep_without_callback_does_not_raise(
    orchestrator: MultiRunOrchestrator,
    spec_a: RunSpec,
) -> None:
    """run_sweep() with on_sweep_complete=None completes without error.

    Args:
        orchestrator: Orchestrator fixture.
        spec_a: Single child sweep.
    """
    result = orchestrator.run_sweep(
        children=[spec_a],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )
    assert len(result.children) == 1


def test_run_sweep_fires_on_run_created_for_parent(
    orchestrator_with_hooks: MultiRunOrchestrator,
    recorded_run_creations: list[RunCreatedEvent],
    spec_a: RunSpec,
    spec_b: RunSpec,
) -> None:
    """run_sweep() fires on_run_created once for its own parent run.

    Children no longer fire an event at the orchestrator level: each child's
    own train()/optimize()/converge() call fires its own on_run_created
    through the forwarded hooks — not exercised by this mocked-execute() test.

    Args:
        orchestrator_with_hooks: Orchestrator fixture wired with hooks.
        recorded_run_creations: Events recorded by the hooks fixture.
        spec_a: First child spec.
        spec_b: Second child spec.
    """
    orchestrator_with_hooks.run_sweep(
        children=[spec_a, spec_b],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )

    assert [event.kind for event in recorded_run_creations] == ["sweep"]
    assert [event.is_outermost for event in recorded_run_creations] == [True]


# ---------------------------------------------------------------------------
# Failure policies
# ---------------------------------------------------------------------------


def test_fail_fast_propagates_child_exception(
    orchestrator: MultiRunOrchestrator,
    mock_execute: MagicMock,
    spec_a: RunSpec,
    spec_b: RunSpec,
) -> None:
    """failure_policy="fail_fast" (default) re-raises a child's exception immediately.

    Args:
        orchestrator: Orchestrator fixture.
        mock_execute: Patched execute() configured to raise on the first call.
        spec_a: First child spec — raises.
        spec_b: Second child spec — must not be reached.
    """
    mock_execute.side_effect = ValueError("boom")

    try:
        orchestrator.run_sweep(
            children=[spec_a, spec_b],
            experiment_name="test_experiment",
            parent_run_name="parent",
        )
        raise AssertionError("expected ValueError to propagate")
    except ValueError as exc:
        assert str(exc) == "boom"

    assert mock_execute.call_count == 1


def test_continue_records_failure_and_runs_later_children(
    orchestrator_with_hooks: MultiRunOrchestrator,
    mock_execute: MagicMock,
    recorded_child_failures: list[ChildFailure],
    spec_a: RunSpec,
    spec_b: RunSpec,
    mock_training_result: TrainingResult,
) -> None:
    """failure_policy="continue" records a ChildFailure and still runs later children.

    Args:
        orchestrator_with_hooks: Orchestrator fixture wired with hooks.
        mock_execute: Patched execute(); first call raises, second succeeds.
        recorded_child_failures: Failures recorded by the hooks fixture.
        spec_a: First child spec — raises.
        spec_b: Second child spec — succeeds.
        mock_training_result: Expected result for the second child.
    """
    mock_execute.side_effect = [ValueError("boom"), mock_training_result]

    result = orchestrator_with_hooks.run_sweep(
        children=[spec_a, spec_b],
        experiment_name="test_experiment",
        parent_run_name="parent",
        failure_policy="continue",
    )

    assert mock_execute.call_count == 2
    first, second = result.children
    assert isinstance(first, ChildFailure)
    assert first.child_id == spec_a.id
    assert first.exception_type == "ValueError"
    assert first.message == "boom"
    assert isinstance(second, ChildSuccess)
    assert second.child_id == spec_b.id

    assert recorded_child_failures == [first]


def test_continue_does_not_tag_parent_as_failed(
    orchestrator: MultiRunOrchestrator,
    mock_execute: MagicMock,
    mock_tracker: MagicMock,
    spec_a: RunSpec,
) -> None:
    """failure_policy="continue" does NOT tag the parent run as failed.

    Args:
        orchestrator: Orchestrator fixture.
        mock_execute: Patched execute(), raises.
        mock_tracker: Tracker mock exposing set_run_tag for inspection.
        spec_a: Single failing child spec.
    """
    mock_execute.side_effect = ValueError("boom")

    orchestrator.run_sweep(
        children=[spec_a],
        experiment_name="test_experiment",
        parent_run_name="parent",
        failure_policy="continue",
    )

    tagged_keys = [call.args[1] for call in mock_tracker.set_run_tag.call_args_list]
    assert "multirun.status" not in tagged_keys


def test_continue_mark_parent_failed_tags_parent_on_any_failure(
    orchestrator: MultiRunOrchestrator,
    mock_execute: MagicMock,
    mock_tracker: MagicMock,
    parent_run_ctx: MagicMock,
    spec_a: RunSpec,
) -> None:
    """failure_policy="continue_mark_parent_failed" tags the parent run once any child fails.

    Args:
        orchestrator: Orchestrator fixture.
        mock_execute: Patched execute(), raises.
        mock_tracker: Tracker mock exposing set_run_tag for inspection.
        parent_run_ctx: Mock parent run context; run_id="parent-run-id".
        spec_a: Single failing child spec.
    """
    mock_execute.side_effect = ValueError("boom")

    result = orchestrator.run_sweep(
        children=[spec_a],
        experiment_name="test_experiment",
        parent_run_name="parent",
        failure_policy="continue_mark_parent_failed",
    )

    assert isinstance(result.children[0], ChildFailure)
    mock_tracker.set_run_tag.assert_called_once_with(
        parent_run_ctx.run_id, "multirun.status", "failed"
    )
