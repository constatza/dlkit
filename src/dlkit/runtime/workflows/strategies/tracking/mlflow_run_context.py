"""Client-based MLflow run context implementation."""

from __future__ import annotations

from typing import Any
from pathlib import Path

from mlflow import MlflowClient

from dlkit.tools.utils.logging_config import get_logger
from .interfaces import IRunContext

logger = get_logger(__name__)


class ClientBasedRunContext(IRunContext):
    """MLflow run context using MlflowClient instead of global functions.

    This implementation eliminates dependency on global MLflow state
    by using explicit client instances for all operations.
    """

    def __init__(self, client: MlflowClient, run_id: str):
        """Initialize run context with client and run ID.

        Args:
            client: MLflow client instance
            run_id: Active run ID for logging operations
        """
        self._client = client
        self._run_id = run_id

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics using MLflow client.

        Args:
            metrics: Dictionary of metric names to values.
            step: Optional step number for the metrics (e.g., epoch number).
        """
        try:
            for key, value in metrics.items():
                self._client.log_metric(self._run_id, key, value, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters using MLflow client."""
        try:
            for key, value in params.items():
                self._client.log_param(self._run_id, key, str(value))
        except Exception as e:
            logger.warning(f"Failed to log params: {e}")

    def log_artifact(self, artifact_path: Path, artifact_dir: str = "") -> None:
        """Log artifact using MLflow client."""
        try:
            self._client.log_artifact(
                self._run_id,
                str(artifact_path),
                artifact_path=artifact_dir or None
            )
        except Exception as e:
            logger.warning(f"Failed to log artifact {artifact_path}: {e}")

    def set_tag(self, key: str, value: str) -> None:
        """Set tag using MLflow client."""
        try:
            self._client.set_tag(self._run_id, key, value)
        except Exception as e:
            logger.warning(f"Failed to set tag {key}: {e}")

    def log_batch(
        self,
        metrics: list[tuple[str, float, int]] | None = None,
        params: list[tuple[str, str]] | None = None,
        tags: list[tuple[str, str]] | None = None,
    ) -> None:
        """Log batch of metrics, params, and tags using MLflow client.

        Args:
            metrics: List of (key, value, step) tuples
            params: List of (key, value) tuples
            tags: List of (key, value) tuples
        """
        try:
            # Convert to MLflow batch format
            from mlflow.entities import Metric, Param, RunTag

            batch_metrics = []
            if metrics:
                for key, value, step in metrics:
                    batch_metrics.append(Metric(key, value, timestamp=None, step=step))

            batch_params = []
            if params:
                for key, value in params:
                    batch_params.append(Param(key, str(value)))

            batch_tags = []
            if tags:
                for key, value in tags:
                    batch_tags.append(RunTag(key, str(value)))

            # Log batch using client
            self._client.log_batch(
                self._run_id,
                metrics=batch_metrics,
                params=batch_params,
                tags=batch_tags,
            )

        except Exception as e:
            logger.warning(f"Failed to log batch: {e}")

    @property
    def run_id(self) -> str:
        """Get current run ID."""
        return self._run_id

    @property
    def client(self) -> MlflowClient:
        """Get MLflow client instance."""
        return self._client