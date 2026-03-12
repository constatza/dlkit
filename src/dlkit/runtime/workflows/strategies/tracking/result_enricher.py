"""Result enrichment service for MLflow tracking.

Single Responsibility: Add MLflow metadata to training results.
"""

from __future__ import annotations

from typing import Any

from dlkit.interfaces.api.domain import TrainingResult
from dlkit.tools.config import GeneralSettings
from dlkit.tools.utils.logging_config import get_logger

from .interfaces import IExperimentTracker, IRunContext

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
        run_context: IRunContext | None = None,
    ) -> TrainingResult:
        """Enrich training result with MLflow tracking metadata.

        Adds MLflow run ID, experiment ID, and tracking URI as both first-class
        fields and metric dict entries (backward compat) without modifying the
        original result.

        Args:
            result: Training result to enrich
            settings: Global settings (currently unused but kept for extensibility)
            tracking_uri: Resolved tracking URI, if available
            run_context: Active run context (preferred source of run_id)

        Returns:
            New TrainingResult with enriched fields and metrics
        """
        if not isinstance(result, TrainingResult):
            logger.warning(f"Cannot enrich non-TrainingResult: {type(result)}")
            return result  # type: ignore[return-value]

        try:
            enriched = dict(getattr(result, "metrics", {}) or {})

            run_id = self._resolve_run_id(run_context)
            resolved_uri = self._resolve_tracking_uri(tracking_uri)

            if run_id:
                enriched["mlflow_run_id"] = run_id
                logger.debug(f"Enriched result with run_id={run_id}")
            else:
                logger.debug("No active MLflow run for enrichment")

            self._add_experiment_id(enriched)

            if resolved_uri:
                enriched["mlflow_tracking_uri"] = resolved_uri

            return TrainingResult(
                model_state=result.model_state,
                metrics=enriched,
                artifacts=result.artifacts,
                duration_seconds=result.duration_seconds,
                predictions=result.predictions,
                mlflow_run_id=run_id,
                mlflow_tracking_uri=resolved_uri,
            )

        except Exception as e:
            logger.warning(f"Failed to enrich result: {e}")
            return result

    def _resolve_run_id(self, run_context: IRunContext | None) -> str | None:
        """Resolve run ID preferring run_context over global state.

        Args:
            run_context: Active run context if available

        Returns:
            Run ID string or None
        """
        if run_context is not None:
            run_id = getattr(run_context, "run_id", None)
            if run_id and run_id != "null-run-id":
                return run_id

        try:
            import mlflow

            run = mlflow.active_run()
            if run:
                return run.info.run_id
        except Exception as e:
            logger.warning(f"Failed to get MLflow run_id from global state: {e}")

        return None

    def _add_experiment_id(self, enriched: dict[str, Any]) -> None:
        """Add MLflow experiment ID to enriched metrics from global state.

        Args:
            enriched: Dictionary to add experiment ID to
        """
        try:
            import mlflow

            run = mlflow.active_run()
            if run:
                enriched["mlflow_experiment_id"] = run.info.experiment_id
        except Exception as e:
            logger.warning(f"Failed to add MLflow experiment_id: {e}")

    def _resolve_tracking_uri(self, tracking_uri: str | None) -> str | None:
        """Resolve tracking URI preferring explicit value over global state.

        Args:
            tracking_uri: Explicit tracking URI if available

        Returns:
            Resolved tracking URI or None
        """
        if tracking_uri:
            return tracking_uri
        try:
            import mlflow

            current_uri = mlflow.get_tracking_uri()
            return current_uri or None
        except Exception as e:
            logger.warning(f"Failed to get MLflow tracking URI: {e}")
            return None
