"""Result enrichment service for MLflow tracking.

Single Responsibility: Add MLflow metadata to training results.
"""

from __future__ import annotations

from dlkit.common import TrainingResult
from dlkit.infrastructure.config import GeneralSettings  # type: ignore
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from dlkit.infrastructure.utils.logging_config import get_logger

from .interfaces import IRunContext

type _WorkflowSettings = GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig

logger = get_logger(__name__)


class ResultEnricher:
    """Enriches training results with MLflow tracking information.

    Single Responsibility: Add MLflow metadata to training results.
    Does not modify the tracker or settings, only enriches result objects.
    """

    def enrich_result(
        self,
        result: TrainingResult,
        settings: _WorkflowSettings,
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
        try:
            enriched = dict(getattr(result, "metrics", {}) or {})

            run_id = self._resolve_run_id(run_context)
            resolved_uri = self._resolve_tracking_uri(tracking_uri, run_context)
            experiment_id = self._resolve_experiment_id(run_context)

            if run_id:
                enriched["mlflow_run_id"] = run_id
                logger.debug("Enriched result with run_id=%s", run_id)
            else:
                logger.debug("No active MLflow run for enrichment")

            if experiment_id:
                enriched["mlflow_experiment_id"] = experiment_id

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
            logger.warning("Failed to enrich result: %s", e)
            return result

    def _resolve_run_id(self, run_context: IRunContext | None) -> str | None:
        """Resolve run ID preferring run_context over global state.

        Args:
            run_context: Active run context if available

        Returns:
            Run ID string or None
        """
        if run_context is not None and run_context.is_active():
            run_id = run_context.run_id
            if run_id:
                return run_id

        return None

    def _resolve_experiment_id(self, run_context: IRunContext | None) -> str | None:
        if run_context is None:
            return None
        return getattr(run_context, "experiment_id", None)

    def _resolve_tracking_uri(
        self,
        tracking_uri: str | None,
        run_context: IRunContext | None,
    ) -> str | None:
        """Resolve tracking URI preferring explicit value over global state.

        Args:
            tracking_uri: Explicit tracking URI if available

        Returns:
            Resolved tracking URI or None
        """
        if tracking_uri:
            return tracking_uri
        if run_context is None:
            return None
        return getattr(run_context, "tracking_uri", None)
