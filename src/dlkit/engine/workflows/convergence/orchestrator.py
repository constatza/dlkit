"""Convergence study orchestrator."""

from __future__ import annotations

import time
from typing import cast

from dlkit.common.errors import WorkflowError
from dlkit.common.results import (
    ChildOutcome,
    ChildSuccess,
    ConvergenceResult,
    TrainingResult,
    WorkflowResult,
)
from dlkit.engine.tracking.interfaces import IRunContext
from dlkit.engine.workflows.multi_run import MultiRunOrchestrator, RunSpec
from dlkit.infrastructure.config.convergence_settings import ConvergenceSettings
from dlkit.infrastructure.config.job_config import ConvergenceJobConfig, TrainingJobConfig

from .aggregation import aggregate_results, build_summary_dict, find_n_star


def _drop_none_values(value: object) -> object:
    """Recursively drop dict keys whose value is None — TOML has no null type.

    ``build_summary_dict``'s ``n_star``/``marginal_gain``/``converged`` fields
    are legitimately ``None`` (e.g. ``marginal_gain`` is always ``None`` for a
    convergence study's first point), so the TOML writer must omit them
    rather than fail to serialise.

    Args:
        value: A dict, list, or scalar to sanitise.

    Returns:
        The same shape with every None-valued dict entry removed.
    """
    if isinstance(value, dict):
        return {k: _drop_none_values(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none_values(v) for v in value]
    return value


def _dict_to_toml(data: dict[str, object]) -> str:
    """Serialise a plain dict to a minimal TOML string using tomlkit.

    Args:
        data: Dictionary with primitive or list values. None values are
            dropped (TOML has no null type) rather than failing to serialise.

    Returns:
        TOML-formatted string.
    """
    from tomlkit import document, dumps, item

    doc = document()
    for key, value in cast("dict[str, object]", _drop_none_values(data)).items():
        if isinstance(value, list):
            arr = item(value)
            doc.add(key, arr)
        else:
            doc.add(key, value)
    return dumps(doc)


def _unwrap_training_result(outcome: ChildOutcome[WorkflowResult]) -> TrainingResult:
    """Type-narrow a multirun child outcome to its TrainingResult.

    Convergence always builds TrainingJobConfig children and always runs the
    sweep under failure_policy="fail_fast", so every outcome is guaranteed a
    successful TrainingResult in practice — but fail loudly instead of
    assuming, in case that guarantee is ever violated.

    Args:
        outcome: One child's outcome from a convergence sweep.

    Returns:
        The wrapped TrainingResult.

    Raises:
        WorkflowError: The outcome is not a successful TrainingResult.
    """
    match outcome:
        case ChildSuccess(result=TrainingResult() as result):
            return result
        case _:
            raise WorkflowError(
                f"Expected a successful TrainingResult from convergence child, got {outcome!r}"
            )


class ConvergenceOrchestrator:
    """Orchestrates a sample-size convergence study using a MultiRunOrchestrator.

    Builds (n × r) RunSpecs, delegates sweep execution to MultiRunOrchestrator,
    logs a TOML summary artifact to the parent run, and returns a ConvergenceResult.

    Args:
        multi_run: Pre-configured MultiRunOrchestrator for sweep execution.
    """

    def __init__(self, multi_run: MultiRunOrchestrator) -> None:
        """Initialise with a pre-configured multi-run orchestrator.

        Args:
            multi_run: MultiRunOrchestrator that owns the MLflow lifecycle.
        """
        self._multi_run = multi_run

    def execute(self, settings: ConvergenceJobConfig) -> ConvergenceResult:
        """Execute the full convergence study.

        Args:
            settings: Validated convergence job configuration.

        Returns:
            ConvergenceResult with points, n_star, and tracking metadata.
        """
        start = time.time()
        sizes = settings.convergence.resolved_sizes()
        children = self._build_children(settings)

        experiment_name = (
            settings.experiment.name if settings.experiment is not None else "convergence"
        )
        parent_run_name = (
            settings.experiment.run_name if settings.experiment is not None else "convergence_sweep"
        )

        def on_sweep_complete(
            parent_run: IRunContext,
            outcomes: tuple[ChildOutcome[WorkflowResult], ...],
        ) -> None:
            results = tuple(_unwrap_training_result(outcome) for outcome in outcomes)
            self._log_summary(parent_run, results, sizes, settings.convergence)

        result = self._multi_run.run_sweep(
            children=children,
            experiment_name=experiment_name,
            parent_run_name=parent_run_name or "convergence_sweep",
            parent_tags={"workflow": "convergence"},
            failure_policy="fail_fast",
            on_sweep_complete=on_sweep_complete,
        )

        training_results = tuple(_unwrap_training_result(outcome) for outcome in result.children)
        points = aggregate_results(training_results, sizes, settings.convergence)
        n_star = find_n_star(points)

        return ConvergenceResult(
            points=points,
            n_star=n_star,
            duration_seconds=time.time() - start,
            mlflow_run_id=result.parent_run_id,
            mlflow_tracking_uri=result.tracking_uri,
        )

    def _build_children(self, settings: ConvergenceJobConfig) -> list[RunSpec]:
        """Build (n × r) RunSpec list from convergence settings.

        Args:
            settings: Convergence job configuration.

        Returns:
            Ordered list of RunSpecs, one per (size, repeat) pair.
        """
        seed = settings.run.seed or 0
        sizes = settings.convergence.resolved_sizes()
        children: list[RunSpec] = []

        for n in sizes:
            for r in range(settings.convergence.repeats):
                patched = settings.patch(
                    {
                        "data.splits.max_train_samples": n,
                        "data.splits.train_subset_seed": seed + r,
                    }
                )
                train_cfg = TrainingJobConfig(
                    run=patched.run,
                    experiment=patched.experiment,
                    model=patched.model,
                    data=patched.data,
                    training=patched.training,
                    tracking=patched.tracking,
                )
                run_id = f"n={n}_r={r}"
                children.append(
                    RunSpec(
                        id=run_id,
                        label=run_id,
                        settings=train_cfg,
                        run_name=run_id,
                        tags={"convergence.n": str(n), "convergence.repeat": str(r)},
                    )
                )

        return children

    def _log_summary(
        self,
        parent_run: IRunContext,
        results: tuple[TrainingResult, ...],
        sizes: tuple[int, ...],
        cfg: ConvergenceSettings,
    ) -> None:
        """Log convergence summary as a TOML artifact and step metrics.

        Args:
            parent_run: Active parent run context.
            results: All training results from the sweep.
            sizes: Ordered sample sizes evaluated.
            cfg: Convergence settings used for aggregation.
        """
        points = aggregate_results(results, sizes, cfg)
        n_star = find_n_star(points)
        summary = build_summary_dict(points, n_star)
        toml_text = _dict_to_toml(summary)
        parent_run.log_artifact_content(toml_text, "convergence_results.toml")

        for point in points:
            parent_run.log_metrics(
                {
                    "convergence/val_mean": point.val_mean,
                    "convergence/gap": point.gap,
                },
                step=point.n,
            )
