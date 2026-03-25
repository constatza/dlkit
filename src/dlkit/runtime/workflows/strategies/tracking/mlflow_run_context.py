"""Client-based MLflow run context implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlflow import MlflowClient

from dlkit.tools.utils.logging_config import get_logger

from .interfaces import IRunContext

logger = get_logger(__name__)


class ClientBasedRunContext(IRunContext):
    """MLflow run context using MlflowClient instead of global functions.

    This implementation eliminates dependency on global MLflow state
    by using explicit client instances for all operations.
    """

    def __init__(self, client: MlflowClient, run_id: str, *, tracking_uri: str):
        """Initialize run context with client, run ID, and tracking URI.

        Args:
            client: MLflow client instance
            run_id: Active run ID for logging operations
            tracking_uri: Tracking URI for this run (for documentation/debugging)
        """
        self._client = client
        self._run_id = run_id
        self._tracking_uri = tracking_uri

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
            logger.warning("Failed to log metrics: %s", e)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters using MLflow client."""
        try:
            for key, value in params.items():
                self._client.log_param(self._run_id, key, str(value))
        except Exception as e:
            logger.warning("Failed to log params: %s", e)

    def log_text(self, text: str, artifact_file: str) -> None:
        """Log text content directly as an artifact without writing to disk.

        Args:
            text: Text content to log.
            artifact_file: Destination path within the run artifact store.
        """
        try:
            self._client.log_text(self._run_id, text, artifact_file)
        except Exception as e:
            logger.warning("Failed to log text artifact '%s': %s", artifact_file, e)

    def log_artifact(self, artifact_path: Path, artifact_dir: str = "") -> None:
        """Log artifact using MLflow client."""
        try:
            self._client.log_artifact(
                self._run_id, str(artifact_path), artifact_path=artifact_dir or None
            )
        except Exception as e:
            logger.warning("Failed to log artifact %s: %s", artifact_path, e)

    def set_tag(self, key: str, value: str) -> None:
        """Set tag using MLflow client."""
        try:
            self._client.set_tag(self._run_id, key, value)
        except Exception as e:
            logger.warning("Failed to set tag %s: %s", key, e)

    def log_dataset(
        self, dataset: Any, context: str | None = None, tags: dict[str, str] | None = None
    ) -> None:
        """Log dataset using MLflow client.

        Args:
            dataset: MLflow dataset object (from mlflow.data.from_numpy, etc.)
            context: Optional context string (e.g., "training", "validation")
            tags: Optional dictionary of tags to associate with the dataset
        """
        try:
            from mlflow.entities import DatasetInput, InputTag

            # Client API expects mlflow.entities.Dataset (not mlflow.data.Dataset wrapper).
            dataset_entity = (
                dataset._to_mlflow_entity() if hasattr(dataset, "_to_mlflow_entity") else dataset
            )

            input_tags = [InputTag(key=k, value=v) for k, v in (tags or {}).items()]
            if context:
                input_tags.append(InputTag(key="mlflow.data.context", value=context))

            dataset_input = DatasetInput(
                dataset=dataset_entity,
                tags=input_tags,
            )

            self._client.log_inputs(self._run_id, datasets=[dataset_input])

        except Exception as e:
            logger.warning("Failed to log dataset: %s", e)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        *,
        registered_model_name: str | None = None,
        signature: Any | None = None,
        input_example: Any | None = None,
    ) -> str | None:
        """Log model artifact with automatic MLflow flavor dispatch."""
        try:
            import mlflow

            kwargs = {
                "name": artifact_path,
                "registered_model_name": registered_model_name,
                "signature": signature,
                "input_example": input_example,
            }

            logged_model = None
            match _resolve_model_flavor(model):
                case "sklearn":
                    logged_model = mlflow.sklearn.log_model(
                        sk_model=model,
                        **kwargs,
                    )
                case "pytorch":
                    logged_model = mlflow.pytorch.log_model(
                        pytorch_model=model,
                        **kwargs,
                    )
                case _:
                    raise TypeError(
                        f"Unsupported model type for MLflow logging: {type(model).__name__}"
                    )

            return _resolve_logged_model_uri(
                logged_model=logged_model,
                fallback_uri=f"runs:/{self._run_id}/{artifact_path}",
            )
        except Exception as e:
            logger.warning("Failed to log model: %s", e)
            return None

    def get_latest_model_version(
        self,
        model_name: str,
        *,
        run_id: str | None = None,
        artifact_path: str | None = None,
    ) -> int | None:
        """Get latest registered model version by numeric max.

        When ``run_id`` is provided, only versions produced by that run are
        considered. This prevents cross-run races when multiple runs register
        new versions around the same time.
        """
        try:
            versions = self._client.search_model_versions(f"name='{model_name}'")
            numeric_versions = [
                int(version.version)
                for version in versions
                if _is_matching_version(
                    version=version,
                    run_id=run_id,
                    artifact_path=artifact_path,
                )
            ]
            return max(numeric_versions) if numeric_versions else None
        except Exception as e:
            logger.warning("Failed to get latest model version for %s: %s", model_name, e)
            return None

    def set_model_alias(self, model_name: str, alias: str, version: int) -> None:
        """Set alias for a registered model version."""
        try:
            self._client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=str(version),
            )
        except Exception as e:
            logger.warning(
                "Failed to set alias '%s' for model '%s' v%s: %s", alias, model_name, version, e
            )

    def set_model_version_tag(
        self,
        model_name: str,
        version: int,
        key: str,
        value: str,
    ) -> None:
        """Set a registered model version tag."""
        try:
            self._client.set_model_version_tag(
                name=model_name,
                version=str(version),
                key=key,
                value=value,
            )
        except Exception as e:
            logger.warning(
                "Failed to set model version tag '%s' for model '%s' v%s: %s",
                key,
                model_name,
                version,
                e,
            )

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
            logger.warning("Failed to log batch: %s", e)

    @property
    def run_id(self) -> str:
        """Get current run ID."""
        return self._run_id

    @property
    def client(self) -> MlflowClient:
        """Get MLflow client instance."""
        return self._client


def _is_sklearn_estimator(model: Any) -> bool:
    """Check sklearn estimator support without hard dependency at import time."""
    try:
        from sklearn.base import BaseEstimator

        return isinstance(model, BaseEstimator)
    except Exception:
        return False


def _resolve_model_flavor(model: Any) -> str:
    if _is_sklearn_estimator(model):
        return "sklearn"

    try:
        import torch

        if isinstance(model, torch.nn.Module):
            return "pytorch"
    except Exception:
        return "unknown"

    return "unknown"


def _resolve_logged_model_uri(logged_model: Any, fallback_uri: str) -> str:
    """Resolve best available MLflow model URI from flavor logging result."""
    model_uri = getattr(logged_model, "model_uri", None)
    if isinstance(model_uri, str) and model_uri:
        return model_uri

    if isinstance(logged_model, str) and logged_model:
        return logged_model

    return fallback_uri


def _is_matching_version(
    *,
    version: Any,
    run_id: str | None,
    artifact_path: str | None,
) -> bool:
    version_value = getattr(version, "version", None)
    if version_value is None:
        return False

    if run_id is not None:
        version_run_id = getattr(version, "run_id", None)
        if version_run_id not in (None, "", run_id):
            return False

    if artifact_path:
        source = str(getattr(version, "source", "") or "")
        normalized_path = f"/{artifact_path.strip('/')}"
        if normalized_path not in source:
            return False

    return True
