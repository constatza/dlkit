"""Result enrichment service for MLflow tracking.

Single Responsibility: Add MLflow metadata to training results.
"""

from __future__ import annotations

from typing import Any

from dlkit.interfaces.api.domain import TrainingResult
from dlkit.tools.config import GeneralSettings
from dlkit.tools.utils.logging_config import get_logger

from .config_accessor import ConfigAccessor
from .interfaces import IExperimentTracker

logger = get_logger(__name__)


class ResultEnricher:
    """Enriches training results with MLflow tracking information.

    Single Responsibility: Add MLflow metadata to training results.
    Does not modify the tracker or settings, only enriches result objects.

    Args:
        tracker: Experiment tracker implementation
    """

    def __init__(self, tracker: IExperimentTracker):
        """Initialize with experiment tracker.

        Args:
            tracker: Experiment tracker implementation
        """
        self._tracker = tracker

    def enrich_result(
        self,
        result: TrainingResult,
        settings: GeneralSettings,
        tracking_uri: str | None = None,
    ) -> TrainingResult:
        """Enrich training result with MLflow tracking metadata.

        Adds MLflow run ID, experiment ID, and tracking URI
        to the result metrics without modifying the original result.

        Args:
            result: Training result to enrich
            settings: Global settings (currently unused but kept for extensibility)
            tracking_uri: Resolved tracking URI, if available

        Returns:
            New TrainingResult with enriched metrics
        """
        if not isinstance(result, TrainingResult):
            logger.warning(f"Cannot enrich non-TrainingResult: {type(result)}")
            return result

        try:
            # Start with existing metrics
            enriched = dict(getattr(result, "metrics", {}) or {})

            # Add MLflow run information from global state
            # Note: Client-based enrichment would require access to run_context,
            # which is not available after the context manager exits.
            # The global MLflow approach is simpler and sufficient for most use cases.
            self._add_mlflow_run_info(enriched)

            # Add tracking URI (prefer resolved URI, fall back to active MLflow URI)
            self._add_tracking_uri(enriched, tracking_uri)

            # Create new result with enriched metrics (preserve predictions)
            return TrainingResult(
                model_state=result.model_state,
                metrics=enriched,
                artifacts=result.artifacts,
                duration_seconds=result.duration_seconds,
                predictions=result.predictions,
            )

        except Exception as e:
            logger.warning(f"Failed to enrich result: {e}")
            return result

    def _add_mlflow_run_info(self, enriched: dict[str, Any]) -> None:
        """Add MLflow run ID and experiment ID to enriched metrics.

        Args:
            enriched: Dictionary to add run info to
        """
        try:
            import mlflow

            run = mlflow.active_run()
            if run:
                enriched["mlflow_run_id"] = run.info.run_id
                enriched["mlflow_experiment_id"] = run.info.experiment_id
                logger.debug(f"Enriched result with run_id={run.info.run_id}")
            else:
                logger.debug("No active MLflow run for enrichment")

        except Exception as e:
            logger.warning(f"Failed to add MLflow run info: {e}")

    def _add_tracking_uri(self, enriched: dict[str, Any], tracking_uri: str | None) -> None:
        """Add MLflow tracking URI to enriched metrics.

        Prefers the resolved tracking URI, falls back to current MLflow URI.

        Args:
            enriched: Dictionary to add tracking URI to
            tracking_uri: Resolved tracking URI from setup (preferred)
        """
        try:
            if tracking_uri:
                enriched["mlflow_tracking_uri"] = tracking_uri
            else:
                import mlflow

                current_uri = mlflow.get_tracking_uri()
                if current_uri:
                    enriched["mlflow_tracking_uri"] = current_uri

        except Exception as e:
            logger.warning(f"Failed to add tracking URI: {e}")
