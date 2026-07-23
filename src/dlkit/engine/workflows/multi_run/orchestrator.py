"""Generic multi-run sweep orchestrator — reusable across convergence, ensemble, ablation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from dlkit.common.hooks import LifecycleHooks, RunCreatedEvent
from dlkit.common.results import (
    ChildFailure,
    ChildOutcome,
    ChildSuccess,
    ConvergenceResult,
    FailurePolicy,
    MultiRunResult,
    OptimizationResult,
    TrainingResult,
    WorkflowResult,
)
from dlkit.engine.tracking.mlflow_tracker import MLflowTracker
from dlkit.engine.workflows.factories.tracking_flag import apply_mlflow_flag

from .spec import RunSpec

if TYPE_CHECKING:
    from dlkit.engine.tracking.interfaces import IRunContext

# One child's dispatch: routes settings to whichever workflow entrypoint
# applies to its concrete type (train/optimize/converge) and returns that
# workflow's result. In practice this is
# `dlkit.engine.workflows.entrypoints.execute`, injected by the caller
# (e.g. `entrypoints/convergence.py`) rather than imported here: `tach.toml`
# only allows `entrypoints -> engine.workflows` (not the reverse), so this
# package cannot import `entrypoints` itself. Called as
# ``dispatch(settings, hooks=hooks)``.
type ChildDispatcher = Callable[..., WorkflowResult]

# MLflow tag applied to a child run, after the fact, pointing back at its
# sweep's parent run. Children are NOT real MLflow-nested runs (each child's
# own train()/optimize()/converge() call opens its own top-level run), so
# this tag is what keeps them discoverable under the sweep (e.g. via
# find_child_run_ids()).
_PARENT_RUN_ID_TAG = "mlflow.parentRunId"

# Tag applied to the parent run when failure_policy="continue_mark_parent_failed"
# and at least one child failed. MLflow has no run-status API surfaced here
# (confirmed: no set_status method on IRunContext/MLflowTracker) — a tag is
# the existing, sufficient mechanism.
_PARENT_FAILED_TAG_KEY = "multirun.status"
_PARENT_FAILED_TAG_VALUE = "failed"


@runtime_checkable
class IMultiRunOrchestrator(Protocol):
    """Protocol for multi-run sweep executors.

    Callers supply child run specifications and (optionally) result
    aggregation; the executor owns the parent MLflow run lifecycle and
    dispatches each child to whichever workflow entrypoint applies to its
    settings type (train/optimize/converge).
    """

    def run_sweep(
        self,
        children: Sequence[RunSpec],
        experiment_name: str,
        parent_run_name: str,
        parent_tags: dict[str, str] | None = None,
        failure_policy: FailurePolicy = "fail_fast",
        on_sweep_complete: Callable[[IRunContext, tuple[ChildOutcome[WorkflowResult], ...]], None]
        | None = None,
    ) -> MultiRunResult[ChildOutcome[WorkflowResult]]:
        """Execute children under a shared parent MLflow run.

        Args:
            children: Ordered sequence of child run specifications.
            experiment_name: MLflow experiment name shared by the parent and
                every child run.
            parent_run_name: MLflow name for the parent sweep run.
            parent_tags: Optional tags for the parent run.
            failure_policy: How to react when a child raises. ``"fail_fast"``
                (default) propagates immediately, matching pre-multirun
                behavior. ``"continue"`` records a
                :class:`~dlkit.common.results.ChildFailure` and proceeds to
                the next child. ``"continue_mark_parent_failed"`` does the
                same and additionally tags the parent run once the sweep
                completes, if any child failed.
            on_sweep_complete: Called with ``(parent_run, outcomes)`` before
                the parent run closes. Use for summary artifact logging.

        Returns:
            MultiRunResult with the parent run id, tracking URI, and one
            ChildOutcome per child, in input order.
        """
        ...


class MultiRunOrchestrator:
    """Opens a parent MLflow run and executes each RunSpec as an independent run.

    Reusable across convergence studies, ensemble runs, and ablation sweeps,
    and across heterogeneous workflow types within a single sweep: each
    child is routed through the injected ``dispatch`` collaborator (in
    practice, ``dlkit.engine.workflows.entrypoints.execute``, which owns that
    child's build/train/track lifecycle end-to-end, including its own
    top-level MLflow run). Dispatch is injected rather than imported because
    ``tach.toml`` only allows ``entrypoints -> engine.workflows``, not the
    reverse — the caller wiring this orchestrator together (which lives in
    ``entrypoints`` or above) supplies it. This orchestrator's own job is the
    parent run's lifecycle, post-hoc child-run tagging so children stay
    discoverable under the parent, and failure-policy bookkeeping.

    Args:
        tracker: MLflow tracker for parent-run management and post-hoc
            child-run tagging.
        dispatch: Routes one child's settings to its workflow result, e.g.
            ``dlkit.engine.workflows.entrypoints.execute``.
        hooks: Optional lifecycle hooks fired around the parent run and
            forwarded into every child's own workflow execution.
    """

    def __init__(
        self,
        tracker: MLflowTracker,
        dispatch: ChildDispatcher,
        hooks: LifecycleHooks | None = None,
    ) -> None:
        self._tracker = tracker
        self._dispatch = dispatch
        self._hooks = hooks

    def run_sweep(
        self,
        children: Sequence[RunSpec],
        experiment_name: str,
        parent_run_name: str,
        parent_tags: dict[str, str] | None = None,
        failure_policy: FailurePolicy = "fail_fast",
        on_sweep_complete: Callable[[IRunContext, tuple[ChildOutcome[WorkflowResult], ...]], None]
        | None = None,
    ) -> MultiRunResult[ChildOutcome[WorkflowResult]]:
        """Execute all children under a shared parent MLflow run.

        Args:
            children: Ordered child run specifications.
            experiment_name: MLflow experiment name.
            parent_run_name: Name for the parent sweep run.
            parent_tags: Tags for the parent run.
            failure_policy: How to react when a child raises; see the
                :class:`IMultiRunOrchestrator` protocol docstring.
            on_sweep_complete: Called with ``(parent_run, outcomes)`` before
                the parent run closes.

        Returns:
            MultiRunResult with the parent run id, tracking URI, and one
            ChildOutcome per child, in input order.

        Note:
            The parent run is created, then immediately closed, before any
            child runs — MLflow's fluent API tracks "the active run" as
            process-global state, not per-tracker-instance, so holding the
            parent open while each child opens its own independent top-level
            run (via the injected ``dispatch``, which builds its own tracker
            from the child's own settings) would collide with a "run already
            active" error. Every parent-run interaction after creation
            (failure tagging, ``on_sweep_complete``) therefore goes through
            post-hoc, non-active-requiring tracker methods
            (``set_run_tag``/``get_run_context``), the same mechanism used
            for post-hoc child-run tagging.
        """
        with self._tracker:
            with self._tracker.create_run(
                experiment_name=experiment_name,
                run_name=parent_run_name,
                tags=parent_tags or {},
            ) as parent_run:
                parent_run_id = parent_run.run_id
                tracking_uri = self._tracker.get_tracking_uri()
                if self._hooks and self._hooks.on_run_created:
                    self._hooks.on_run_created(
                        RunCreatedEvent(
                            run_id=parent_run_id,
                            tracking_uri=tracking_uri,
                            kind="sweep",
                            is_outermost=True,
                        )
                    )

            # parent_run is now closed (not globally active) — children can
            # each safely open their own independent top-level run below.
            outcomes = tuple(
                self._run_one(spec, experiment_name, parent_run_id, failure_policy)
                for spec in children
            )

            if failure_policy == "continue_mark_parent_failed" and any(
                isinstance(outcome, ChildFailure) for outcome in outcomes
            ):
                self._tracker.set_run_tag(
                    parent_run_id, _PARENT_FAILED_TAG_KEY, _PARENT_FAILED_TAG_VALUE
                )

            if on_sweep_complete is not None:
                on_sweep_complete(self._tracker.get_run_context(parent_run_id), outcomes)

            return MultiRunResult(
                parent_run_id=parent_run_id,
                tracking_uri=tracking_uri,
                children=outcomes,
            )

    def _run_one(
        self,
        spec: RunSpec,
        experiment_name: str,
        parent_run_id: str,
        failure_policy: FailurePolicy,
    ) -> ChildOutcome[WorkflowResult]:
        """Execute one child via the injected dispatcher.

        Patches the child's settings so its run lands in the sweep's own
        experiment (mirroring the pre-multi_run-package behavior of forcing
        every child into the parent's experiment) and under ``spec.run_name``,
        then ensures MLflow tracking is enabled before dispatching. On
        success, tags the child's own MLflow run with the parent run id so
        it stays discoverable under the sweep.

        Args:
            spec: The child run specification to execute.
            experiment_name: MLflow experiment name shared with the parent run.
            parent_run_id: The sweep's parent run id, used for post-hoc
                child-run tagging.
            failure_policy: Governs whether an exception propagates
                (``"fail_fast"``) or is captured as a ``ChildFailure``
                (``"continue"``/``"continue_mark_parent_failed"``).

        Returns:
            ChildSuccess wrapping the workflow result, or ChildFailure if the
            child raised under a continuing failure policy.

        Raises:
            Exception: Whatever the dispatched workflow raises, when
                ``failure_policy == "fail_fast"``.
        """
        child_settings = apply_mlflow_flag(
            spec.settings.patch(
                {"experiment": {"name": experiment_name, "run_name": spec.run_name}}
            ),
            mlflow=True,
        )

        try:
            result = self._dispatch(child_settings, hooks=self._hooks)
        except Exception as exc:
            if failure_policy == "fail_fast":
                raise
            failure = ChildFailure(
                child_id=spec.id,
                exception_type=type(exc).__name__,
                message=str(exc),
                # Can't reliably recover a partial run id from a failed
                # dispatch() call — best-effort None rather than guessing.
                run_id=None,
                stage="execute",
            )
            if self._hooks and self._hooks.on_child_failed:
                self._hooks.on_child_failed(failure)
            return failure

        run_id = _extract_run_id(result)
        if run_id is not None:
            self._tracker.set_run_tag(run_id, _PARENT_RUN_ID_TAG, parent_run_id)

        return ChildSuccess(child_id=spec.id, run_id=run_id, result=result)


def _extract_run_id(result: WorkflowResult) -> str | None:
    """Best-effort MLflow run id for a child's workflow result.

    All three ``WorkflowResult`` variants (``TrainingResult``,
    ``ConvergenceResult``, ``OptimizationResult``) carry ``mlflow_run_id``.

    Args:
        result: The child's workflow result.

    Returns:
        The MLflow run id if the result type carries one, else None.
    """
    match result:
        case (
            TrainingResult(mlflow_run_id=run_id)
            | ConvergenceResult(mlflow_run_id=run_id)
            | OptimizationResult(mlflow_run_id=run_id)
        ):
            return run_id
        case _:
            return None
