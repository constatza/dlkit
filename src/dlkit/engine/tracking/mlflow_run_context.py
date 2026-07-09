"""Client-based MLflow run context implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

from mlflow import MlflowClient

from dlkit.common.hooks import ParamValue
from dlkit.infrastructure.utils.logging_config import get_logger

from .best_effort import best_effort
from .binary_artifact import log_binary_artifact
from .interfaces import IRunContext, LoggedDataset, LoggedModel

if TYPE_CHECKING:
    from mlflow.entities import Dataset as MlflowDatasetEntity
    from mlflow.models import ModelSignature

logger = get_logger(__name__)


def _resolve_dataset_entity(dataset: LoggedDataset) -> MlflowDatasetEntity:
    conversion = getattr(dataset, "_to_mlflow_entity", None)
    if not callable(conversion):
        return cast("MlflowDatasetEntity", dataset)
    return conversion()


class ClientBasedRunContext(IRunContext):
    """MLflow run context using MlflowClient instead of global functions.

    This implementation eliminates dependency on global MLflow state
    by using explicit client instances for all operations.
    """

    def __init__(
        self,
        client: MlflowClient,
        run_id: str,
        *,
        tracking_uri: str,
        experiment_id: str | None = None,
    ):
        """Initialize run context with client, run ID, and tracking URI.

        Args:
            client: MLflow client instance
            run_id: Active run ID for logging operations
            tracking_uri: Tracking URI for this run (for documentation/debugging)
        """
        self._client = client
        self._run_id = run_id
        self._tracking_uri = tracking_uri
        self._experiment_id = experiment_id

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def experiment_id(self) -> str | None:
        return self._experiment_id

    @property
    def tracking_uri(self) -> str | None:
        return self._tracking_uri

    @best_effort("log metrics")
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics using MLflow client.

        Args:
            metrics: Dictionary of metric names to values.
            step: Optional step number for the metrics (e.g., epoch number).
        """
        for key, value in metrics.items():
            self._client.log_metric(self._run_id, key, value, step=step)

    @best_effort("log params")
    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        """Log parameters using MLflow client."""
        for key, value in params.items():
            self._client.log_param(self._run_id, key, str(value))

    @best_effort("log artifact content")
    def log_artifact_content(self, content: str | bytes, artifact_file: str) -> None:
        """Log artifact content directly without writing a tracked temp file.

        Args:
            content: Artifact content to log.
            artifact_file: Destination path within the run artifact store.
        """
        if isinstance(content, bytes):
            log_binary_artifact(self._client, self._run_id, content, artifact_file)
            return
        self._client.log_text(self._run_id, content, artifact_file)

    @best_effort("log artifact")
    def log_artifact(self, artifact_path: Path, artifact_dir: str = "") -> None:
        """Log artifact using MLflow client."""
        self._client.log_artifact(
            self._run_id, str(artifact_path), artifact_path=artifact_dir or None
        )

    @best_effort("set tag")
    def set_tag(self, key: str, value: str) -> None:
        """Set tag using MLflow client."""
        self._client.set_tag(self._run_id, key, value)

    @best_effort("log dataset")
    def log_dataset(
        self,
        dataset: LoggedDataset,
        context: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Log dataset using MLflow client.

        Args:
            dataset: MLflow dataset object (from mlflow.data.from_numpy, etc.)
            context: Optional context string (e.g., "training", "validation")
            tags: Optional dictionary of tags to associate with the dataset
        """
        from mlflow.entities import DatasetInput, InputTag

        # Client API expects mlflow.entities.Dataset (not mlflow.data.Dataset wrapper).
        dataset_entity = _resolve_dataset_entity(dataset)

        input_tags = [InputTag(key=k, value=v) for k, v in (tags or {}).items()]
        if context:
            input_tags.append(InputTag(key="mlflow.data.context", value=context))

        dataset_input = DatasetInput(
            dataset=dataset_entity,
            tags=input_tags,
        )

        self._client.log_inputs(self._run_id, datasets=[dataset_input])

    def log_model(
        self,
        model: LoggedModel,
        artifact_path: str,
        *,
        registered_model_name: str | None = None,
        signature: ModelSignature | None = None,
        # Intentionally opaque foreign backend payload forwarded to MLflow.
        input_example: object | None = None,
        model_serialization_format: str | None = None,
    ) -> str | None:
        """Log model artifact with automatic MLflow flavor dispatch.

        Raises on failure — callers must not silently proceed without the artifact.
        """
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
                logged_model = mlflow.sklearn.log_model(sk_model=model, **kwargs)
            case "pytorch":
                if model_serialization_format is not None:
                    kwargs["serialization_format"] = model_serialization_format
                logged_model = mlflow.pytorch.log_model(pytorch_model=model, **kwargs)
            case _:
                raise TypeError(
                    f"Unsupported model type for MLflow logging: {type(model).__name__}"
                )

        return _resolve_logged_model_uri(
            logged_model=logged_model,
            fallback_uri=f"runs:/{self._run_id}/{artifact_path}",
        )

    @best_effort("log batch")
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

    @property
    def client(self) -> MlflowClient:
        """Get MLflow client instance."""
        return self._client


def _is_sklearn_estimator(model: object) -> bool:
    """Check sklearn estimator support without hard dependency at import time."""
    try:
        from sklearn.base import BaseEstimator

        return isinstance(model, BaseEstimator)
    except Exception:
        return False


def _resolve_model_flavor(model: object) -> str:
    if _is_sklearn_estimator(model):
        return "sklearn"

    try:
        import torch
    except Exception:
        return "unknown"

    if isinstance(model, torch.nn.Module):
        return "pytorch"
    return "unknown"


def _resolve_logged_model_uri(logged_model: object, fallback_uri: str) -> str:
    """Resolve best available MLflow model URI from flavor logging result."""
    model_uri = getattr(logged_model, "model_uri", None)
    if isinstance(model_uri, str) and model_uri:
        return model_uri

    if isinstance(logged_model, str) and logged_model:
        return logged_model

    return fallback_uri
