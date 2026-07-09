"""Metric logging service for MLflow tracking.

Single Responsibility: Log training/validation/test metrics to experiment tracker.
"""

from __future__ import annotations

import math
from typing import Any

from dlkit.common import TrainingResult
from dlkit.common.metric_stages import STAGE_ALIASES
from dlkit.engine.artifacts import IMetricSink
from dlkit.infrastructure.utils.logging_config import get_logger

logger = get_logger(__name__)

# Prefixes/suffixes of metric keys already logged, with epoch steps, by
# MLflowEpochLogger. Test metrics are excluded on purpose: they are a single
# scalar over the whole test set, not an epoch-indexed series, so they get
# logged exactly once here as a final, step-less value.
EPOCH_LOGGED_PREFIXES: tuple[str, ...] = tuple(
    f"{alias}{sep}"
    for stage in ("train", "val")
    for alias in STAGE_ALIASES[stage]
    for sep in ("_", "/")
)
EPOCH_LOGGED_SUFFIXES: tuple[str, ...] = tuple(
    f" {alias}" for stage in ("train", "val") for alias in STAGE_ALIASES[stage]
)


def split_stage_filtered_metrics(
    metrics: dict[str, Any],
    prefixes: tuple[str, ...] = EPOCH_LOGGED_PREFIXES,
    suffixes: tuple[str, ...] = EPOCH_LOGGED_SUFFIXES,
) -> tuple[dict[str, float], dict[str, str]]:
    """Split metrics into numeric/fallback dicts, dropping already-epoch-logged keys.

    Args:
        metrics: Raw metric name -> value mapping.
        prefixes: Key prefixes (lowercase) to exclude as already epoch-logged.
        suffixes: Key suffixes (lowercase) to exclude as already epoch-logged.

    Returns:
        Tuple of (numeric metrics, non-numeric metrics coerced to strings).
    """
    numeric: dict[str, float] = {}
    fallback: dict[str, str] = {}

    for key, value in metrics.items():
        key_lower = key.lower()
        if any(key_lower.startswith(prefix) for prefix in prefixes):
            continue
        if any(key_lower.endswith(suffix) for suffix in suffixes):
            continue

        try:
            numeric_value = float(value)
            if math.isnan(numeric_value) or math.isinf(numeric_value):
                raise ValueError("non-finite metric")
            numeric[key] = numeric_value
        except TypeError, ValueError:
            if value is not None:
                fallback[f"metric_{key}"] = str(value)

    return numeric, fallback


class MetricLogger:
    """Logs training/validation/test metrics to a run context.

    Separates numeric metrics from non-numeric fallbacks.
    Stateless — all context is passed per-call via ``run_context``.
    """

    def log_summary_metrics(
        self,
        result: TrainingResult,
        run_context: IMetricSink,
    ) -> None:
        """Log summary metrics excluding stage-specific metrics already logged by callbacks.

        MLflowEpochLogger callback logs train/val metrics during execution with epoch
        steps. This method logs everything else — including test metrics, which are a
        single scalar over the whole test set and therefore logged once here with no
        step, instead of at both an epoch step and step 0.

        Args:
            result: Training result containing metrics
            run_context: Run context for logging
        """
        metrics = getattr(result, "metrics", None)
        if not metrics:
            return

        numeric, fallback = split_stage_filtered_metrics(metrics)

        if numeric:
            run_context.log_metrics(numeric)
        if fallback:
            run_context.log_params(fallback)

    def log_all_metrics(
        self,
        result: TrainingResult,
        run_context: IMetricSink,
    ) -> None:
        """Log all scalar metrics from the training result to the tracker, unfiltered.

        Unlike log_summary_metrics, this logs ALL metrics without filtering.
        Use this when epoch logger is not available or for complete metric capture.

        Args:
            result: Training result containing metrics
            run_context: Run context for logging
        """
        metrics = getattr(result, "metrics", None)
        if not metrics:
            return

        numeric, fallback = split_stage_filtered_metrics(metrics, prefixes=(), suffixes=())

        if numeric:
            run_context.log_metrics(numeric)
        if fallback:
            run_context.log_params(fallback)
