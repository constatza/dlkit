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
        server_url: str | None = None,
        server_status: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """Enrich training result with MLflow tracking metadata.

        Adds MLflow run ID, experiment ID, tracking URI, and server information
        to the result metrics without modifying the original result.

        Args:
            result: Training result to enrich
            settings: Global settings (currently unused but kept for extensibility)
            server_url: MLflow server URL if available
            server_status: MLflow server status if available

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

            # Add tracking URI (prefer server URL, fall back to client URI)
            self._add_tracking_uri(enriched, server_url)

            # Add server metadata
            self._add_server_metadata(enriched, server_url, server_status)

            # Create new result with enriched metrics
            return TrainingResult(
                model_state=result.model_state,
                metrics=enriched,
                artifacts=result.artifacts,
                duration_seconds=result.duration_seconds,
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

    def _add_tracking_uri(self, enriched: dict[str, Any], server_url: str | None) -> None:
        """Add MLflow tracking URI to enriched metrics.

        Prefers server URL from setup, falls back to client's tracking URI.

        Args:
            enriched: Dictionary to add tracking URI to
            server_url: Server URL from setup (preferred)
        """
        try:
            if server_url:
                enriched["mlflow_tracking_uri"] = server_url
            else:
                # Fall back to client's tracking URI if tracker supports it
                try:
                    if hasattr(self._tracker, "get_client"):
                        client = self._tracker.get_client()  # type: ignore[attr-defined]
                        tracking_uri = getattr(client, "tracking_uri", None)
                        if tracking_uri:
                            enriched["mlflow_tracking_uri"] = tracking_uri
                except Exception as e:
                    logger.debug(f"Could not get tracking URI from client: {e}")

        except Exception as e:
            logger.warning(f"Failed to add tracking URI: {e}")

    def _add_server_metadata(
        self,
        enriched: dict[str, Any],
        server_url: str | None,
        server_status: dict[str, Any] | None,
    ) -> None:
        """Add MLflow server metadata to enriched metrics.

        Args:
            enriched: Dictionary to add server metadata to
            server_url: Server URL if available
            server_status: Server status if available
        """
        try:
            if server_url:
                enriched["mlflow_server_url"] = server_url

            if server_status is not None:
                enriched["mlflow_server_running"] = bool(server_status.get("running"))
                response_time = server_status.get("response_time")
                if response_time is not None:
                    enriched["mlflow_server_response_time"] = response_time

        except Exception as e:
            logger.warning(f"Failed to add server metadata: {e}")
