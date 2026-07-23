"""Tests for MultiRunOrchestrator.run_sweep().

Covers:
- MultiRunOrchestrator satisfies IMultiRunOrchestrator protocol
- run_sweep() calls entrypoints.execute() exactly once per child, with hooks forwarded
- run_sweep() patches each child's settings to the sweep's experiment/run_name
- run_sweep() merges spec.tags into the child's experiment.tags
- run_sweep() returns a MultiRunResult with one ChildSuccess per child, in order
- run_sweep() calls on_sweep_complete exactly once with (parent_run_ctx, outcomes)
- run_sweep() without a callback does not raise
- run_sweep() fires on_run_created once for its own parent run (kind='sweep')
- run_sweep() tags a successful child's own MLflow run with the parent run id
- a ChildFailure carries the child's best-effort run_id when on_run_created fired first
- hooks.on_sweep_complete (LifecycleHooks) fires with a SweepCompletedEvent
- failure_policy="fail_fast" (default) propagates a child's exception immediately
- failure_policy="continue" records a ChildFailure and still runs later children
- failure_policy="continue_mark_parent_failed" additionally tags the parent run
- ChildSuccess/ChildFailure carry label/params/metadata from their RunSpec
- on_child_planned fires once per child, before dispatch, with the correct payload
- on_child_completed fires with the exact ChildSuccess on success, not on failure
"""

from __future__ import annotations

from unittest.mock import MagicMock

from dlkit.common.hooks import (
    ChildPlannedEvent,
    LifecycleHooks,
    RunCreatedEvent,
    SweepCompletedEvent,
)
from dlkit.common.results import (
    ChildFailure,
    ChildSuccess,
    EvaluationResult,
    MultiRunResult,
    TrainingResult,
    WorkflowResult,
)
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
    recorded_run_creations: list[RunCreatedEvent],
    spec_a: RunSpec,
) -> None:
    """run_sweep() forwards its own hooks into each execute() call.

    The forwarded hooks object wraps ``on_run_created`` (to capture a
    best-effort child run_id for ChildFailure — see
    test_continue_records_best_effort_run_id_on_failure), so it is not the
    same object as the orchestrator's own ``hooks``. Every other hook, and
    the *behavior* of ``on_run_created`` (still recording the event), must
    be preserved.

    Args:
        orchestrator_with_hooks: Orchestrator fixture wired with hooks.
        mock_execute: Patched execute() to inspect call kwargs.
        hooks: The same LifecycleHooks instance orchestrator_with_hooks was built with.
        recorded_run_creations: Events recorded by the hooks fixture's on_run_created.
        spec_a: Single child spec.
    """
    orchestrator_with_hooks.run_sweep(
        children=[spec_a],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )
    forwarded_hooks = mock_execute.call_args.kwargs["hooks"]
    assert forwarded_hooks.on_child_failed is hooks.on_child_failed

    event = RunCreatedEvent(
        run_id="child-run-id", tracking_uri=None, kind="train", is_outermost=True
    )
    forwarded_hooks.on_run_created(event)
    assert recorded_run_creations[-1] == event


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


def test_run_one_merges_spec_tags_into_experiment_tags(
    orchestrator: MultiRunOrchestrator,
    mock_execute: MagicMock,
    spec_with_tags: RunSpec,
) -> None:
    """_run_one() merges spec.tags into the child's own experiment.tags.

    Regression guard for tag-filtered run lookup: a caller's assignment_id /
    dataset_id / job_id tags must actually reach the child's MLflow run, not
    just sit unused on RunSpec.

    Args:
        orchestrator: Orchestrator fixture.
        mock_execute: Patched execute() to inspect the settings it was called with.
        spec_with_tags: Single child spec carrying tags={"assignment_id": "42", ...}.
    """
    orchestrator.run_sweep(
        children=[spec_with_tags],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )
    called_settings = mock_execute.call_args.args[0]
    assert called_settings.experiment.tags == spec_with_tags.tags


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


def test_run_sweep_extracts_and_tags_run_id_for_evaluation_result(
    orchestrator: MultiRunOrchestrator,
    mock_tracker: MagicMock,
    mock_execute: MagicMock,
    spec_a: RunSpec,
) -> None:
    """A dispatched child returning EvaluationResult (evaluate as a multirun
    child) has its run_id extracted and tagged exactly like the other three
    WorkflowResult variants — the ``_extract_run_id`` match arm added for
    ``EvaluationResult``.
    """
    evaluation_result = EvaluationResult(
        predictions=None,
        targets=None,
        metrics={"mae": 0.1},
        figures={},
        duration_seconds=1.0,
        mlflow_run_id="eval-child-run-id",
    )
    mock_execute.return_value = evaluation_result

    result = orchestrator.run_sweep(
        children=[spec_a],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )

    assert len(result.children) == 1
    outcome = result.children[0]
    assert isinstance(outcome, ChildSuccess)
    assert outcome.run_id == "eval-child-run-id"
    assert outcome.result is evaluation_result
    mock_tracker.set_run_tag.assert_called_once_with(
        "eval-child-run-id", "mlflow.parentRunId", "parent-run-id"
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


def test_run_sweep_fires_hooks_on_sweep_complete(
    mock_tracker: MagicMock,
    mock_execute: MagicMock,
    spec_a: RunSpec,
    spec_b: RunSpec,
) -> None:
    """hooks.on_sweep_complete fires once with a SweepCompletedEvent.

    Regression guard for the public multirun API's on_sweep_complete gap:
    LifecycleHooks.on_sweep_complete must actually be reachable from
    run_sweep(), not just accepted and ignored.

    Args:
        mock_tracker: Tracker mock fixture; parent run_id="parent-run-id".
        mock_execute: Mock dispatcher fixture.
        spec_a: First child spec.
        spec_b: Second child spec.
    """
    recorded: list[SweepCompletedEvent] = []
    hooks_with_sweep_complete = LifecycleHooks(on_sweep_complete=recorded.append)
    orchestrator = MultiRunOrchestrator(
        tracker=mock_tracker, dispatch=mock_execute, hooks=hooks_with_sweep_complete
    )

    orchestrator.run_sweep(
        children=[spec_a, spec_b],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )

    assert len(recorded) == 1
    event = recorded[0]
    assert event.run_id == "parent-run-id"
    assert len(event.outcomes) == 2


def test_continue_records_best_effort_run_id_on_failure(
    orchestrator_with_hooks: MultiRunOrchestrator,
    mock_execute: MagicMock,
    spec_a: RunSpec,
) -> None:
    """A ChildFailure carries the child's run_id if it opened one before failing.

    Regression guard for the "worse than today" discoverability gap: a
    child that opens its own MLflow run and then crashes must not become an
    untagged, unreferenced orphan run — the orchestrator captures the run_id
    via the child's own on_run_created firing (forwarded through hooks)
    before dispatch() raises.

    Args:
        orchestrator_with_hooks: Orchestrator fixture wired with hooks.
        mock_execute: Patched execute(); fires on_run_created then raises.
        spec_a: Single failing child spec.
    """

    def _fire_then_raise(settings: object, *, hooks: LifecycleHooks) -> None:
        assert hooks.on_run_created is not None
        hooks.on_run_created(
            RunCreatedEvent(
                run_id="orphaned-child-run", tracking_uri=None, kind="train", is_outermost=True
            )
        )
        raise ValueError("boom")

    mock_execute.side_effect = _fire_then_raise

    result = orchestrator_with_hooks.run_sweep(
        children=[spec_a],
        experiment_name="test_experiment",
        parent_run_name="parent",
        failure_policy="continue",
    )

    failure = result.children[0]
    assert isinstance(failure, ChildFailure)
    assert failure.run_id == "orphaned-child-run"


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
# label/params/metadata propagation
# ---------------------------------------------------------------------------


def test_child_success_carries_label_params_metadata(
    orchestrator: MultiRunOrchestrator,
    spec_with_tags: RunSpec,
) -> None:
    """ChildSuccess carries label/params/metadata matching the input RunSpec.

    Args:
        orchestrator: Orchestrator fixture.
        spec_with_tags: Single child spec with params={"lr": 0.01},
            metadata={"note": "x"}.
    """
    result = orchestrator.run_sweep(
        children=[spec_with_tags],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )

    success = result.children[0]
    assert isinstance(success, ChildSuccess)
    assert success.label == spec_with_tags.label
    assert success.params == spec_with_tags.params
    assert success.metadata == spec_with_tags.metadata


def test_child_failure_carries_label_params_metadata(
    orchestrator: MultiRunOrchestrator,
    mock_execute: MagicMock,
    spec_with_tags: RunSpec,
) -> None:
    """ChildFailure carries label/params/metadata from its RunSpec when the child raises.

    Args:
        orchestrator: Orchestrator fixture.
        mock_execute: Patched execute(), raises.
        spec_with_tags: Single failing child spec with params={"lr": 0.01},
            metadata={"note": "x"}.
    """
    mock_execute.side_effect = ValueError("boom")

    result = orchestrator.run_sweep(
        children=[spec_with_tags],
        experiment_name="test_experiment",
        parent_run_name="parent",
        failure_policy="continue",
    )

    failure = result.children[0]
    assert isinstance(failure, ChildFailure)
    assert failure.label == spec_with_tags.label
    assert failure.params == spec_with_tags.params
    assert failure.metadata == spec_with_tags.metadata


# ---------------------------------------------------------------------------
# on_child_planned / on_child_completed hooks
# ---------------------------------------------------------------------------


def test_on_child_planned_fires_once_per_child_with_correct_payload(
    orchestrator_with_hooks: MultiRunOrchestrator,
    recorded_child_planned: list[ChildPlannedEvent],
    spec_a: RunSpec,
    spec_b: RunSpec,
) -> None:
    """on_child_planned fires exactly once per child with the correct payload.

    Args:
        orchestrator_with_hooks: Orchestrator fixture wired with hooks.
        recorded_child_planned: Events recorded by the hooks fixture.
        spec_a: First child spec.
        spec_b: Second child spec.
    """
    orchestrator_with_hooks.run_sweep(
        children=[spec_a, spec_b],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )

    assert len(recorded_child_planned) == 2
    for event, spec in zip(recorded_child_planned, [spec_a, spec_b], strict=True):
        assert event.child_id == spec.id
        assert event.label == spec.label
        assert event.run_name == spec.run_name
        assert event.tags == spec.tags
        assert event.params == spec.params
        assert event.metadata == spec.metadata


def test_on_child_planned_fires_before_dispatch(
    orchestrator_with_hooks: MultiRunOrchestrator,
    mock_execute: MagicMock,
    recorded_child_planned: list[ChildPlannedEvent],
    spec_a: RunSpec,
) -> None:
    """on_child_planned fires before the child is dispatched.

    mock_execute's side_effect asserts the planned event was already
    recorded by the time execute() is called, proving dispatch-order rather
    than merely that both fire at some point.

    Args:
        orchestrator_with_hooks: Orchestrator fixture wired with hooks.
        mock_execute: Patched execute(); asserts on_child_planned already fired.
        recorded_child_planned: Events recorded by the hooks fixture.
        spec_a: Single child spec.
    """

    def _assert_planned_already_recorded(
        settings: object, *, hooks: LifecycleHooks
    ) -> TrainingResult:
        assert len(recorded_child_planned) == 1
        return TrainingResult(model_state=None, metrics={}, artifacts={}, duration_seconds=0.1)

    mock_execute.side_effect = _assert_planned_already_recorded

    orchestrator_with_hooks.run_sweep(
        children=[spec_a],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )

    assert len(recorded_child_planned) == 1


def test_on_child_completed_fires_with_returned_child_success(
    orchestrator_with_hooks: MultiRunOrchestrator,
    recorded_child_completed: list[ChildSuccess[WorkflowResult]],
    spec_a: RunSpec,
) -> None:
    """on_child_completed fires on success with the exact ChildSuccess returned in the result.

    Args:
        orchestrator_with_hooks: Orchestrator fixture wired with hooks.
        recorded_child_completed: Successes recorded by the hooks fixture.
        spec_a: Single child spec.
    """
    result = orchestrator_with_hooks.run_sweep(
        children=[spec_a],
        experiment_name="test_experiment",
        parent_run_name="parent",
    )

    assert len(recorded_child_completed) == 1
    assert recorded_child_completed[0] is result.children[0]


def test_on_child_completed_does_not_fire_for_failing_child(
    orchestrator_with_hooks: MultiRunOrchestrator,
    mock_execute: MagicMock,
    recorded_child_completed: list[ChildSuccess[WorkflowResult]],
    recorded_child_failures: list[ChildFailure],
    spec_a: RunSpec,
) -> None:
    """on_child_completed does not fire for a failing child; only on_child_failed does.

    Args:
        orchestrator_with_hooks: Orchestrator fixture wired with hooks.
        mock_execute: Patched execute(), raises.
        recorded_child_completed: Successes recorded by the hooks fixture.
        recorded_child_failures: Failures recorded by the hooks fixture.
        spec_a: Single failing child spec.
    """
    mock_execute.side_effect = ValueError("boom")

    orchestrator_with_hooks.run_sweep(
        children=[spec_a],
        experiment_name="test_experiment",
        parent_run_name="parent",
        failure_policy="continue",
    )

    assert recorded_child_completed == []
    assert len(recorded_child_failures) == 1


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
