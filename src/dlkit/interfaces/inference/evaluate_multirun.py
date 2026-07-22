"""Batch evaluation of every child run under a multirun/sweep parent.

Distinct from ``evaluate()`` (single checkpoint) — this fans a single
``evaluate()`` call out over every child run of a parent MLflow run,
matching on the ``mlflow.parentRunId`` tag convention so it works for both
dlkit-native nested sweeps (``MultiRunOrchestrator``) and externally-linked
runs sharing the same convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dlkit.common import EvaluationResult, MultiRunResult
from dlkit.common.checkpoint_source import RunCheckpoint
from dlkit.common.hooks import LifecycleHooks
from dlkit.engine.tracking.run_queries import find_child_run_ids
from dlkit.infrastructure.config.job_config import InferenceJobConfig
from dlkit.infrastructure.config.plot_settings import PlotSettings

from .evaluate import evaluate


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildEvaluation:
    """One child run's evaluation, keyed by the run it came from.

    A plain ``EvaluationResult`` isn't enough on its own here: its own
    ``mlflow_run_id`` field (when set) names the run created to *log this
    eval result* (only present when ``log_to_mlflow=True``), not the run the
    checkpoint was pulled from — a different concept entirely. ``run_id``
    disambiguates by naming the source checkpoint's run explicitly.
    """

    run_id: str
    result: EvaluationResult


def evaluate_multirun(
    settings: InferenceJobConfig,
    parent_run_id: str,
    *,
    split: Literal["test", "predict"] = "test",
    plots: PlotSettings | None = None,
    log_to_mlflow: bool = False,
    hooks: LifecycleHooks | None = None,
    device: str = "auto",
    batch_size: int = 32,
) -> MultiRunResult[ChildEvaluation]:
    """Evaluate every child run of a multirun/sweep parent run.

    Args:
        settings: Inference job configuration shared across all children.
        parent_run_id: MLflow run id of the multirun/sweep parent. Works for
            both dlkit-native nested sweeps (``MultiRunOrchestrator``) and
            externally-linked runs sharing the same ``mlflow.parentRunId``
            tag convention.
        split: Which labeled split to evaluate against, forwarded to every
            child ``evaluate()`` call.
        plots: Plot configuration, forwarded to every child ``evaluate()``
            call.
        log_to_mlflow: If True, each child evaluation opens its own MLflow
            run and logs metrics + figures.
        hooks: Optional lifecycle hooks, forwarded to every child
            ``evaluate()`` call.
        device: Inference device, forwarded to every child ``evaluate()``
            call.
        batch_size: Dataloader batch size, forwarded to every child
            ``evaluate()`` call.

    Returns:
        MultiRunResult keyed by ``parent_run_id``, with one ChildEvaluation
        per active child run, in ascending ``start_time`` order.

    Raises:
        WorkflowError: ``parent_run_id`` does not exist or has no child
            runs.
    """
    child_run_ids = find_child_run_ids(parent_run_id=parent_run_id)
    children = tuple(
        ChildEvaluation(
            run_id=child_run_id,
            result=evaluate(
                settings,
                run_checkpoint=RunCheckpoint(run_id=child_run_id),
                split=split,
                plots=plots,
                log_to_mlflow=log_to_mlflow,
                hooks=hooks,
                device=device,
                batch_size=batch_size,
            ),
        )
        for child_run_id in child_run_ids
    )
    return MultiRunResult(parent_run_id=parent_run_id, children=children)
