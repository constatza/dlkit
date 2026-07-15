"""Eval-only orchestration: checkpoint + labeled dataset -> metrics + figures.

Trainer-free counterpart to the training-time plot callbacks. Reuses the same
figure generators (via ``select_enabled_generators``) and the same
``IArtifactLogger``/``IRunContext`` artifact-logging contracts as training —
no new abstractions, no Lightning ``Trainer``, no weight updates.
"""

from __future__ import annotations

import io
import time
from typing import TYPE_CHECKING, Literal

from dlkit.common.results import EvaluationResult
from dlkit.engine.adapters.lightning.plot_callbacks import select_enabled_generators
from dlkit.infrastructure.utils.logging_config import get_logger

from .batch_prediction import run_batched_evaluation

if TYPE_CHECKING:
    from lightning.pytorch import LightningDataModule
    from matplotlib.figure import Figure
    from torch import Tensor

    from dlkit.engine.tracking.interfaces import IRunContext
    from dlkit.infrastructure.config.plot_settings import PlotSettings

    from .predictor import CheckpointPredictor

_log = get_logger(__name__)


def compute_regression_metrics(predictions: Tensor, targets: Tensor) -> dict[str, float]:
    """Compute MAE / RMSE / R2 over flattened predictions and targets.

    Single-target metrics only, matching ``predictor.predict_target_key`` —
    no routing/aggregation across multiple targets.

    Args:
        predictions: Prediction tensor, any shape.
        targets: Ground-truth tensor, same total elements as predictions.

    Returns:
        Dict with keys ``"mae"``, ``"rmse"``, ``"r2"``.
    """
    from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

    preds_flat = predictions.reshape(-1).float()
    targets_flat = targets.reshape(-1).float()
    return {
        "mae": float(MeanAbsoluteError()(preds_flat, targets_flat)),
        "rmse": float(MeanSquaredError(squared=False)(preds_flat, targets_flat)),
        "r2": float(R2Score()(preds_flat, targets_flat)),
    }


def generate_regression_figures(
    predictions: Tensor,
    targets: Tensor,
    plots: PlotSettings,
) -> dict[str, Figure]:
    """Generate the regression figures enabled by ``plots``.

    Calls ``select_enabled_generators(plots)`` — the same selection used by
    the training-time ``PredictionPlotCallback`` — directly on flattened
    numpy arrays. No callback machinery, no MLflow dependency.

    Args:
        predictions: Prediction tensor, any shape.
        targets: Ground-truth tensor, same total elements as predictions.
        plots: Plot configuration controlling which figures are generated.

    Returns:
        Dict mapping each generator's ``name`` to its rendered Figure. Figures
        are returned open; the caller decides whether to display, save, or
        close them.
    """
    preds_flat = predictions.detach().cpu().numpy().reshape(-1)
    tgts_flat = targets.detach().cpu().numpy().reshape(-1)
    return {
        gen.name: gen.generate(preds_flat, tgts_flat) for gen in select_enabled_generators(plots)
    }


def evaluate_checkpoint(
    predictor: CheckpointPredictor,
    datamodule: LightningDataModule,
    plots: PlotSettings,
    split: Literal["test", "predict"] = "test",
) -> EvaluationResult:
    """Evaluate a loaded predictor against a labeled dataset split.

    No MLflow logging — pairs with ``log_evaluation_result`` for that. Never
    constructs a Lightning ``Trainer``.

    Args:
        predictor: Loaded predictor to run inference with.
        datamodule: Datamodule providing the requested split's dataloader.
        plots: Plot configuration controlling which figures are generated.
        split: Which labeled split to evaluate against.

    Returns:
        EvaluationResult with predictions, targets, metrics, and figures.
    """
    start = time.perf_counter()
    predictions, targets = run_batched_evaluation(predictor, datamodule, split=split)
    metrics = compute_regression_metrics(predictions, targets)
    figures = generate_regression_figures(predictions, targets, plots)
    duration_seconds = time.perf_counter() - start

    return EvaluationResult(
        predictions=predictions.detach().cpu().numpy(),
        targets=targets.detach().cpu().numpy(),
        metrics=metrics,
        figures=figures,
        duration_seconds=duration_seconds,
    )


def _figure_to_bytes(fig: Figure, plots: PlotSettings) -> bytes:
    """Encode a Figure to bytes in-memory without closing it.

    Args:
        fig: matplotlib Figure to encode.
        plots: Plot configuration for format and dpi.

    Returns:
        Encoded image bytes.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=plots.format, dpi=plots.dpi, bbox_inches="tight")
    return buf.getvalue()


def log_evaluation_result(
    result: EvaluationResult,
    run_context: IRunContext,
    plots: PlotSettings,
) -> None:
    """Log an EvaluationResult's metrics and figures to an active tracking run.

    Unlike the training-time ``_plot_and_log`` helper, this does NOT close the
    figures afterward — ``result.figures`` still holds references that the
    caller may want to display or save locally.

    Args:
        result: Evaluation result to log.
        run_context: Active tracking run context (or any ``IArtifactLogger``).
        plots: Plot configuration for artifact format, dpi, and directory.
    """
    run_context.log_metrics(result.metrics)
    for name, fig in result.figures.items():
        try:
            content = _figure_to_bytes(fig, plots)
            run_context.log_artifact_content(content, f"{plots.artifact_dir}/{name}.{plots.format}")
        except Exception as exc:
            _log.warning("log_evaluation_result: failed to save/upload '{}' — {}", name, exc)


__all__ = [
    "compute_regression_metrics",
    "evaluate_checkpoint",
    "generate_regression_figures",
    "log_evaluation_result",
]
