"""Generic multi-run sweep orchestrator — reusable across convergence, ensemble, ablation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from dlkit.common import TrainingResult
from dlkit.engine.tracking.mlflow_tracker import MLflowTracker
from dlkit.engine.training.interfaces import ITrainingExecutor
from dlkit.engine.workflows.factories.build_factory import BuildFactory
from dlkit.engine.workflows.factories.build_strategy import WorkflowSettings

if TYPE_CHECKING:
    from dlkit.engine.artifacts import IMetricSink
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
    """

    def __init__(
        self,
        build_factory: BuildFactory,
        executor: ITrainingExecutor,
        tracker: MLflowTracker,
    ) -> None:
        self._build_factory = build_factory
        self._executor = executor
        self._tracker = tracker

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
            components = self._build_factory.build_components(variant.settings)
            self._inject_epoch_logger(components, child_run)
            return self._executor.execute(components, variant.settings)

    @staticmethod
    def _inject_epoch_logger(components: object, run_context: IMetricSink) -> None:
        """Append MLflowEpochLogger to trainer callbacks before training starts.

        Mirrors TrialExecutor._inject_mlflow_logger() from the HPO workflow.

        Args:
            components: RuntimeComponents with a mutable trainer.callbacks list.
            run_context: IRunContext used as the metric sink.
        """
        from dlkit.engine.adapters.lightning.callbacks import MLflowEpochLogger

        trainer = getattr(components, "trainer", None)
        if not trainer:
            return
        if not hasattr(trainer, "callbacks"):
            trainer.callbacks = []
        trainer.callbacks.append(MLflowEpochLogger(run_context))
