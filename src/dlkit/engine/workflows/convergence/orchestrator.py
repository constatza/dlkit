"""Convergence study orchestrator."""

from __future__ import annotations

import time

from dlkit.common.results import ConvergenceResult, TrainingResult
from dlkit.engine.tracking.interfaces import IRunContext
from dlkit.engine.workflows.multi_run import MultiRunOrchestrator, RunVariant
from dlkit.infrastructure.config.convergence_settings import ConvergenceSettings
from dlkit.infrastructure.config.job_config import ConvergenceJobConfig, TrainingJobConfig

from .aggregation import aggregate_results, build_summary_dict, find_n_star


def _dict_to_toml(data: dict[str, object]) -> str:
    """Serialise a plain dict to a minimal TOML string using tomlkit.

    Args:
        data: Dictionary with primitive or list values.

    Returns:
        TOML-formatted string.
    """
    from tomlkit import document, dumps, item

    doc = document()
    for key, value in data.items():
        if isinstance(value, list):
            arr = item(value)
            doc.add(key, arr)
        else:
            doc.add(key, value)
    return dumps(doc)


class ConvergenceOrchestrator:
    """Orchestrates a sample-size convergence study using a MultiRunOrchestrator.

    Builds (n × r) RunVariants, delegates sweep execution to MultiRunOrchestrator,
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
        variants = self._build_variants(settings)

        experiment_name = (
            settings.experiment.name if settings.experiment is not None else "convergence"
        )
        parent_run_name = (
            settings.experiment.run_name if settings.experiment is not None else "convergence_sweep"
        )

        run_id: str | None = None
        tracking_uri: str | None = None

        def on_sweep_complete(
            parent_run: IRunContext,
            results: tuple[TrainingResult, ...],
        ) -> None:
            nonlocal run_id, tracking_uri
            run_id = parent_run.run_id
            tracking_uri = parent_run.tracking_uri
            self._log_summary(parent_run, results, sizes, settings.convergence)

        results = self._multi_run.run_sweep(
            variants=variants,
            experiment_name=experiment_name,
            parent_run_name=parent_run_name or "convergence_sweep",
            parent_tags={"workflow": "convergence"},
            on_sweep_complete=on_sweep_complete,
        )

        points = aggregate_results(results, sizes, settings.convergence)
        n_star = find_n_star(points)

        return ConvergenceResult(
            points=points,
            n_star=n_star,
            duration_seconds=time.time() - start,
            mlflow_run_id=run_id,
            mlflow_tracking_uri=tracking_uri,
        )

    def _build_variants(self, settings: ConvergenceJobConfig) -> list[RunVariant]:
        """Build (n × r) RunVariant list from convergence settings.

        Args:
            settings: Convergence job configuration.

        Returns:
            Ordered list of RunVariants, one per (size, repeat) pair.
        """
        seed = settings.run.seed or 0
        sizes = settings.convergence.resolved_sizes()
        variants: list[RunVariant] = []

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
                variants.append(
                    RunVariant(
                        settings=train_cfg,
                        run_name=f"n={n}_r={r}",
                        tags={"convergence.n": str(n), "convergence.repeat": str(r)},
                    )
                )

        return variants

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
