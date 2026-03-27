"""Metric logging service for MLflow tracking.

Single Responsibility: Log training/validation/test metrics to experiment tracker.
"""

from __future__ import annotations

import math

from dlkit.domain import TrainingResult
from dlkit.tools.utils.logging_config import get_logger

from .interfaces import IExperimentTracker, IRunContext

logger = get_logger(__name__)


class MetricLogger:
    """Handles metric logging to MLflow.

    Single Responsibility: Log training/validation/test metrics to experiment tracker.
    Separates numeric metrics from non-numeric fallbacks.

    Args:
        tracker: Experiment tracker implementation
    """

    def __init__(self, tracker: IExperimentTracker):
        """Initialize with experiment tracker.

        Args:
            tracker: Experiment tracker implementation
        """
        self._tracker = tracker

    def log_summary_metrics(
        self,
        result: TrainingResult,
        run_context: IRunContext,
    ) -> None:
        """Log summary metrics excluding stage-specific metrics already logged by callbacks.

        MLflowEpochLogger callback logs train/val/test metrics during execution with epoch steps.
        This method logs only additional metrics (like status, custom metrics) that wouldn't
        be captured by the epoch logger, preventing duplicate entries in MLflow.

        Args:
            result: Training result containing metrics
            run_context: Run context for logging
        """
        metrics = getattr(result, "metrics", None)
        if not metrics:
            return

        # Filter out stage-specific metrics that are already logged by MLflowEpochLogger
        # These typically have prefixes like "train_", "val_", "test_" or suffixes like " test"
        stage_prefixes = (
            "train_",
            "train/",
            "val_",
            "val/",
            "valid_",
            "validation_",
            "test_",
            "test/",
        )
        stage_suffixes = (" test", " train", " val", " validation")

        numeric: dict[str, float] = {}
        fallback: dict[str, str] = {}

        for key, value in metrics.items():
            # Skip metrics that are stage-specific (already logged by epoch logger)
            key_lower = key.lower()
            if any(key_lower.startswith(prefix) for prefix in stage_prefixes):
                continue
            if any(key_lower.endswith(suffix) for suffix in stage_suffixes):
                continue

            # Separate numeric from non-numeric metrics
            try:
                numeric_value = float(value)
                if math.isnan(numeric_value) or math.isinf(numeric_value):
                    raise ValueError("non-finite metric")
                numeric[key] = numeric_value
            except TypeError, ValueError:
                if value is not None:
                    fallback[f"metric_{key}"] = str(value)

        # Log to MLflow
        if numeric:
            run_context.log_metrics(numeric)
        if fallback:
            run_context.log_params(fallback)

    def log_all_metrics(
        self,
        result: TrainingResult,
        run_context: IRunContext,
    ) -> None:
        """Log all scalar metrics from the training result to the tracker.

        Unlike log_summary_metrics, this logs ALL metrics without filtering.
        Use this when epoch logger is not available or for complete metric capture.

        Args:
            result: Training result containing metrics
            run_context: Run context for logging
        """
        metrics = getattr(result, "metrics", None)
        if not metrics:
            return

        numeric: dict[str, float] = {}
        fallback: dict[str, str] = {}

        for key, value in metrics.items():
            try:
                numeric_value = float(value)
                if math.isnan(numeric_value) or math.isinf(numeric_value):
                    raise ValueError("non-finite metric")
                numeric[key] = numeric_value
            except TypeError, ValueError:
                if value is not None:
                    fallback[f"metric_{key}"] = str(value)

        if numeric:
            run_context.log_metrics(numeric)
        if fallback:
            run_context.log_params(fallback)
