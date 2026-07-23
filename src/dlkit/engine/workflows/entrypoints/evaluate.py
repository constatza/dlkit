"""Runtime-owned evaluation workflow entrypoint: checkpoint + labeled dataset -> metrics + figures.

Distinct from ``train``/``optimize``/``converge`` (which fit model parameters)
and from ``load_model()``/``predictor.predict()`` (raw predictions only, no
targets/metrics/plots). ``evaluate()`` never constructs a Lightning
``Trainer`` and never updates weights.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import cast

from dlkit.common import ConfigurationError, EvaluationResult
from dlkit.common.checkpoint_source import LatestRunCheckpoint, RunCheckpoint
from dlkit.common.hooks import LifecycleHooks, RunCreatedEvent
from dlkit.engine.inference import (
    evaluate_checkpoint,
    load_model_from_settings,
    log_evaluation_result,
)
from dlkit.engine.tracking.checkpoint_recovery import download_checkpoint_artifact
from dlkit.engine.tracking.mlflow_tracker import MLflowTracker
from dlkit.engine.tracking.run_queries import find_latest_run_id
from dlkit.engine.workflows.factories.inference_data_factory import build_inference_datamodule
from dlkit.infrastructure.config.job_config import InferenceJobConfig
from dlkit.infrastructure.config.plot_settings import PlotSettings

from ._entrypoint_context import EntrypointContext
from ._override_types import EvaluationOverrides, require_override_model

_DEFAULT_EVAL_PLOTS = PlotSettings(
    enabled=True,
    parity=True,
    residual=True,
    error_histogram=True,
    residual_vs_index=True,
)

_DEFAULT_EXPERIMENT_NAME = "dlkit-evaluate"


def _resolve_checkpoint_path(settings: InferenceJobConfig) -> Path | str | None:
    """Resolve ``settings.model.checkpoint`` to a concrete local path.

    Args:
        settings: Inference job configuration whose ``model.checkpoint`` may
            be a literal path or a ``CheckpointSource`` to resolve via MLflow.

    Returns:
        The literal checkpoint path unchanged, or the local path a resolved
        run's checkpoint artifact was downloaded to.
    """
    tracking_uri = settings.tracking.uri
    match settings.model.checkpoint:
        case RunCheckpoint(run_id=run_id):
            resolved_run_id = run_id
        case LatestRunCheckpoint(experiment_name=experiment_name):
            resolved_run_id = find_latest_run_id(
                experiment_name=experiment_name
                or (settings.experiment.name if settings.experiment else _DEFAULT_EXPERIMENT_NAME),
                tracking_uri=tracking_uri,
            )
        case checkpoint_path:
            return checkpoint_path

    return download_checkpoint_artifact(
        resolved_run_id,
        Path(tempfile.mkdtemp(prefix="dlkit-eval-checkpoint-")),
        tracking_uri=tracking_uri,
    )


def evaluate(
    settings: InferenceJobConfig,
    overrides: EvaluationOverrides | None = None,
    *,
    hooks: LifecycleHooks | None = None,
) -> EvaluationResult:
    """Evaluate a trained checkpoint against a labeled dataset split.

    Produces the same regression stats/plots as training (MAE/RMSE/R2, plus
    parity/residual/error-histogram/residual-vs-index figures) by reusing the
    exact figure generators used during training — without training.

    ``apply_transforms`` is always ``True`` internally and not exposed: raw
    dataset targets and inverse-transformed predictions must be compared on
    the same scale, so this is an invariant rather than a caller-facing knob.

    Checkpoint selection, split, device, and MLflow logging are all
    settings-driven rather than function kwargs, matching ``train()``/
    ``optimize()``/``converge()``:

    - ``settings.model.checkpoint``: a literal path, or a ``CheckpointSource``
      (``RunCheckpoint(run_id=...)`` / ``LatestRunCheckpoint(experiment_name=...)``)
      resolved from a previously trained MLflow run. Assign via
      ``settings.patch({"model": {"checkpoint": RunCheckpoint(run_id="abc123")}})``.
    - ``settings.split``: ``"test"`` (default) or ``"predict"``.
    - ``settings.device``: inference device, ``"auto"`` by default.
    - ``settings.tracking.backend == "mlflow"``: opens an MLflow run and logs
      metrics + figures, exactly how ``train()``/``optimize()`` decide to log.
    - ``settings.experiment.run_name``: MLflow run name, if logging.
    - ``settings.data.batch_size``: dataloader batch size for evaluation.
    - ``settings.plots``: plot configuration. Defaults to all four regression
      plots enabled unless ``settings.plots.enabled`` is already set (in
      which case its explicit flags win) — plots are the point of calling
      this function, so they default on here (unlike training, where
      ``PlotSettings`` defaults to opt-in).

    Args:
        settings: Inference job configuration. ``settings.data.targets`` must
            be non-empty so the requested split carries ground truth.
        overrides: Optional runtime overrides (``checkpoint_path``,
            ``experiment_name``, ``run_name``, ``tags``, ``batch_size``,
            ``split``, ``device``).
        hooks: Optional lifecycle hooks. ``on_run_created`` fires immediately
            after the run is created, before any metrics/figures are logged
            — the same extension point ``train()``/``execute()`` use to nest
            a run under an externally created parent, here with
            ``kind="evaluate"`` and ``is_outermost=True`` (evaluate never
            creates nested child runs).

    Returns:
        EvaluationResult with predictions, targets, metrics, and figures.

    Raises:
        ConfigurationError: ``settings.data`` is unset or has no targets.
    """
    validated_overrides = require_override_model(overrides, EvaluationOverrides)
    context = EntrypointContext.prepare(settings, validated_overrides, workflow_name="evaluation")
    settings = cast(InferenceJobConfig, context.settings)

    if settings.data is None or not settings.data.targets:
        raise ConfigurationError(
            "evaluate() requires settings.data.targets to be configured — "
            "there is no ground truth to compare predictions against otherwise."
        )

    resolved_checkpoint_path = _resolve_checkpoint_path(settings)

    resolved_plots = settings.plots if settings.plots.enabled else _DEFAULT_EVAL_PLOTS

    predictor = load_model_from_settings(
        settings,
        checkpoint_path=resolved_checkpoint_path,
        device=settings.device,
        batch_size=settings.data.batch_size,
        apply_transforms=True,
    )
    try:
        datamodule = build_inference_datamodule(
            settings, checkpoint_override=resolved_checkpoint_path
        )
        result = evaluate_checkpoint(predictor, datamodule, resolved_plots, split=settings.split)
    finally:
        predictor.unload()

    if settings.tracking.backend == "mlflow":
        tracker = MLflowTracker()
        tracker.configure(settings.tracking)
        exp_name = settings.experiment.name if settings.experiment else _DEFAULT_EXPERIMENT_NAME
        run_name = settings.experiment.run_name if settings.experiment else None
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
