"""Eval-only API function: checkpoint + labeled dataset -> metrics + figures.

Distinct from ``train``/``optimize``/``converge`` (which fit model parameters)
and from ``load_model()``/``predictor.predict()`` (raw predictions only, no
targets/metrics/plots). ``evaluate()`` never constructs a Lightning
``Trainer`` and never updates weights.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

from dlkit.common import ConfigurationError, EvaluationResult
from dlkit.common.hooks import LifecycleHooks, RunCreatedEvent
from dlkit.engine.inference import (
    evaluate_checkpoint,
    load_model_from_settings,
    log_evaluation_result,
)
from dlkit.engine.tracking.mlflow_tracker import MLflowTracker
from dlkit.engine.workflows.factories.inference_data_factory import build_inference_datamodule
from dlkit.infrastructure.config.job_config import InferenceJobConfig
from dlkit.infrastructure.config.plot_settings import PlotSettings

_DEFAULT_EVAL_PLOTS = PlotSettings(
    enabled=True,
    parity=True,
    residual=True,
    error_histogram=True,
    residual_vs_index=True,
)


def evaluate(
    settings: InferenceJobConfig,
    *,
    checkpoint_path: Path | str | None = None,
    split: Literal["test", "predict"] = "test",
    plots: PlotSettings | None = None,
    log_to_mlflow: bool = False,
    run_name: str | None = None,
    hooks: LifecycleHooks | None = None,
    device: str = "auto",
    batch_size: int = 32,
) -> EvaluationResult:
    """Evaluate a trained checkpoint against a labeled dataset split.

    Produces the same regression stats/plots as training (MAE/RMSE/R2, plus
    parity/residual/error-histogram/residual-vs-index figures) by reusing the
    exact figure generators used during training — without training.

    ``apply_transforms`` is always ``True`` internally and not exposed: raw
    dataset targets and inverse-transformed predictions must be compared on
    the same scale, so this is an invariant rather than a caller-facing knob.

    Args:
        settings: Inference job configuration. ``settings.data.targets`` must
            be non-empty so the requested split carries ground truth.
        checkpoint_path: Checkpoint override; defaults to
            ``settings.model.checkpoint``.
        split: Which labeled split to evaluate against. ``"test"`` (default)
            uses the held-out test partition; ``"predict"`` uses the predict
            partition for datamodules where it also carries labels.
        plots: Plot configuration. Defaults to all four regression plots
            enabled unless ``settings.plots.enabled`` is already set (in
            which case its explicit flags win) — plots are the point of
            calling this function, so they default on here (unlike training,
            where ``PlotSettings`` defaults to opt-in).
        log_to_mlflow: If True, open an MLflow run and log metrics + figures.
        run_name: Optional MLflow run name (only used when ``log_to_mlflow``).
        hooks: Optional lifecycle hooks (only used when ``log_to_mlflow``).
            ``on_run_created`` fires immediately after the run is created,
            before any metrics/figures are logged — the same extension point
            ``train()``/``execute()`` use to nest a run under an externally
            created parent, here with ``kind="evaluate"`` and
            ``is_outermost=True`` (evaluate never creates nested child runs).
        device: Inference device (``"auto"``, ``"cpu"``, ``"cuda"``, ...).
        batch_size: Dataloader batch size for evaluation.

    Returns:
        EvaluationResult with predictions, targets, metrics, and figures.

    Raises:
        ConfigurationError: ``settings.data`` is unset or has no targets.
    """
    if settings.data is None or not settings.data.targets:
        raise ConfigurationError(
            "evaluate() requires settings.data.targets to be configured — "
            "there is no ground truth to compare predictions against otherwise."
        )

    resolved_plots = plots
    if resolved_plots is None:
        resolved_plots = settings.plots if settings.plots.enabled else _DEFAULT_EVAL_PLOTS

    predictor = load_model_from_settings(
        settings,
        checkpoint_path=checkpoint_path,
        device=device,
        batch_size=batch_size,
        apply_transforms=True,
    )
    try:
        datamodule = build_inference_datamodule(settings)
        result = evaluate_checkpoint(predictor, datamodule, resolved_plots, split=split)
    finally:
        predictor.unload()

    if log_to_mlflow:
        tracker = MLflowTracker()
        tracker.configure(settings.tracking)
        exp_name = settings.experiment.name if settings.experiment else "dlkit-evaluate"
        with tracker, tracker.create_run(experiment_name=exp_name, run_name=run_name) as run:
            if hooks is not None and hooks.on_run_created is not None:
                hooks.on_run_created(
                    RunCreatedEvent(
                        run_id=run.run_id,
                        tracking_uri=tracker.get_tracking_uri(),
                        kind="evaluate",
                        is_outermost=True,
                    )
                )
            log_evaluation_result(result, run, resolved_plots)
            result = replace(
                result,
                mlflow_run_id=run.run_id,
                mlflow_tracking_uri=tracker.get_tracking_uri(),
            )

    return result
