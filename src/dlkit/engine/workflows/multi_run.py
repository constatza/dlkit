"""Generic multi-run sweep orchestrator — reusable across convergence, ensemble, ablation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from dlkit.common import TrainingResult
from dlkit.common.hooks import LifecycleHooks
from dlkit.engine.tracking.mlflow_tracker import MLflowTracker
from dlkit.engine.tracking.tracking_decorator import TrackingDecorator
from dlkit.engine.training.interfaces import ITrainingExecutor
from dlkit.engine.workflows.factories.build_factory import BuildFactory
from dlkit.engine.workflows.factories.build_strategy import WorkflowSettings

if TYPE_CHECKING:
    from dlkit.engine.tracking.interfaces import IRunContext


@dataclass(frozen=True)
class RunVariant:
    """A single training run specification within a sweep.

    Args:
        settings: Fully-patched training config for this variant.
        run_name: MLflow child run name, e.g. ``"n=100_r=0"``.
        tags: Optional MLflow tags for this run.
    """

    settings: WorkflowSettings
    run_name: str
    tags: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class IMultiRunOrchestrator(Protocol):
    """Protocol for multi-run sweep executors.

    Callers implement variant generation and result aggregation; the executor
    owns the MLflow lifecycle and per-variant training execution.
    """

    def run_sweep(
        self,
        variants: Sequence[RunVariant],
        experiment_name: str,
        parent_run_name: str,
        parent_tags: dict[str, str] | None = None,
        on_sweep_complete: Callable[[IRunContext, tuple[TrainingResult, ...]], None] | None = None,
    ) -> tuple[TrainingResult, ...]:
        """Execute variants as nested MLflow runs under a shared parent.

        Args:
            variants: Ordered sequence of run specifications.
            experiment_name: MLflow experiment name for the parent run.
            parent_run_name: MLflow name for the parent sweep run.
            parent_tags: Optional tags for the parent run.
            on_sweep_complete: Called with ``(parent_run, results)`` before
                the parent run closes. Use for summary artifact logging.

        Returns:
            TrainingResult for each variant, in input order.
        """
        ...


class MultiRunOrchestrator:
    """Opens a parent MLflow run and executes each RunVariant as a nested child run.

    Reusable across convergence studies, ensemble runs, and ablation sweeps.
    Callers control variant generation and result aggregation; this class owns
    only the MLflow lifecycle and per-variant training execution.

    Args:
        build_factory: Builds RuntimeComponents from settings.
        executor: Executes training given pre-built components.
        tracker: MLflow tracker for parent/child run management.
        hooks: Optional lifecycle hooks fired around each child run.
    """

    def __init__(
        self,
        build_factory: BuildFactory,
        executor: ITrainingExecutor,
        tracker: MLflowTracker,
        hooks: LifecycleHooks | None = None,
    ) -> None:
        self._build_factory = build_factory
        self._tracker = tracker
        self._hooks = hooks
        self._tracking_decorator = TrackingDecorator(executor, tracker, hooks=hooks)

    def run_sweep(
        self,
        variants: Sequence[RunVariant],
        experiment_name: str,
        parent_run_name: str,
        parent_tags: dict[str, str] | None = None,
        on_sweep_complete: Callable[[IRunContext, tuple[TrainingResult, ...]], None] | None = None,
    ) -> tuple[TrainingResult, ...]:
        """Execute all variants as nested MLflow runs under a shared parent.

        Args:
            variants: Ordered run specifications.
            experiment_name: MLflow experiment name.
            parent_run_name: Name for the parent sweep run.
            parent_tags: Tags for the parent run.
            on_sweep_complete: Called with ``(parent_run, results)`` before
                the parent run closes.

        Returns:
            TrainingResult for each variant, in input order.
        """
        with self._tracker:
            with self._tracker.create_run(
                experiment_name=experiment_name,
                run_name=parent_run_name,
                tags=parent_tags or {},
            ) as parent_run:
                results = tuple(self._run_one(v) for v in variants)
                if on_sweep_complete is not None:
                    on_sweep_complete(parent_run, results)
                return results

    def _run_one(self, variant: RunVariant) -> TrainingResult:
        """Execute one variant as a nested child run.

        Fires ``on_run_created`` immediately after opening the child run
        (this method is the one that knows the run was just created), then
        delegates the full train()-parity tracking lifecycle — settings and
        dataset logging, plot/checkpoint callbacks, metric/artifact logging,
        and the remaining LifecycleHooks — to
        :meth:`TrackingDecorator.execute_within_run`.

        Args:
            variant: The run specification to execute.

        Returns:
            TrainingResult from the training executor.
        """
        with self._tracker.create_run(
            run_name=variant.run_name,
            nested=True,
            tags=variant.tags,
        ) as child_run:
            tracking_uri = self._tracker.get_tracking_uri()
            if self._hooks and self._hooks.on_run_created:
                self._hooks.on_run_created(child_run.run_id, tracking_uri)

            components = self._build_factory.build_components(variant.settings)
            return self._tracking_decorator.execute_within_run(
                components,
                variant.settings,
                run_context=child_run,
                tracking_uri=tracking_uri,
            )
