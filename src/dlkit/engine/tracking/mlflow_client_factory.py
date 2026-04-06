"""MLflow client factory for proper client instance management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from mlflow import MlflowClient
from mlflow.environment_variables import (
    MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR,
    MLFLOW_HTTP_REQUEST_BACKOFF_JITTER,
    MLFLOW_HTTP_REQUEST_MAX_RETRIES,
    MLFLOW_HTTP_REQUEST_TIMEOUT,
)

from dlkit.infrastructure.io.url_utils import parse_url
from dlkit.infrastructure.utils.logging_config import get_logger

logger = get_logger(__name__)


class MLflowClientFactory:
    """Factory for creating properly configured MLflow client instances.

    This factory eliminates dependency on global MLflow state by creating
    client instances with explicit tracking URI configuration.
    """

    @staticmethod
    def create_client(
        tracking_uri: str | None = None,
    ) -> MlflowClient:
        """Create MLflow client with explicit configuration.

        Args:
            tracking_uri: Override tracking URI

        Returns:
            Configured MlflowClient instance
        """
        uri = str(tracking_uri) if tracking_uri else None

        # Validate tracking URI if provided
        if uri:
            try:
                parsed = parse_url(uri)
                if not parsed.scheme:
                    logger.warning("Tracking URI missing scheme: %s", uri)
            except Exception as e:
                logger.warning("Invalid tracking URI format: %s - %s", uri, e)

        # Create client with explicit tracking URI
        if uri:
            logger.debug("Creating MLflow client with tracking URI: %s", uri)
            return MlflowClient(tracking_uri=uri)
        logger.debug("Creating MLflow client with default tracking URI")
        return MlflowClient()

    @staticmethod
    def validate_client_connectivity(client: MlflowClient) -> bool:
        """Validate that client can connect to MLflow tracking server."""
        try:
            # Simple connectivity test - try to list experiments with trimmed timeouts
            # Note: All MLflow HTTP env vars must be integers
            with _temporary_request_settings(timeout=3, max_retries=1, backoff_factor=1):
                client.search_experiments()
            return True
        except Exception as e:
            logger.warning("MLflow client connectivity test failed: %s", e)
            return False

    @staticmethod
    def get_or_create_experiment(
        client: MlflowClient,
        experiment_name: str,
        artifact_location: str | None = None,
    ) -> str:
        """Get experiment ID or create if it doesn't exist."""
        logger.debug("Checking if experiment '%s' exists", experiment_name)
        try:
            logger.debug("Calling client.get_experiment_by_name('%s')", experiment_name)
            experiment = client.get_experiment_by_name(experiment_name)
            logger.debug("get_experiment_by_name returned: %s", experiment)
            if experiment is not None:
                logger.debug(f"Experiment exists with ID: {experiment.experiment_id}")
                return experiment.experiment_id
        except Exception as e:
            logger.debug(f"get_experiment_by_name failed with: {type(e).__name__}: {e}")

        # Create new experiment if not found
        logger.debug("Creating new MLflow experiment: %s", experiment_name)
        result = client.create_experiment(experiment_name, artifact_location=artifact_location)
        logger.debug("Experiment created with ID: %s", result)
        return result


@contextmanager
def _temporary_request_settings(
    timeout: int,
    max_retries: int,
    backoff_factor: int,
) -> Iterator[None]:
    """Temporarily override MLflow HTTP request behaviour for fast validation.

    MLflow's defaults favour resiliency (large timeouts and exponential
    backoff). For connectivity checks during our tests this translates into
    multi-minute waits when the endpoint is unreachable. We temporarily tighten
    the relevant environment-driven knobs so the check either succeeds quickly
    or fails within a couple of seconds, and restore the original configuration
    afterwards to avoid side effects for real workloads.

    Note: All MLflow HTTP environment variables must be integers.
    """

    def _swap(env_var, value: int) -> tuple[bool, str | None]:
        was_set = env_var.is_set()
        previous = env_var.get_raw() if was_set else None
        env_var.set(str(value))
        return was_set, previous

    swaps = [
        (MLFLOW_HTTP_REQUEST_TIMEOUT, timeout),
        (MLFLOW_HTTP_REQUEST_MAX_RETRIES, max_retries),
        (MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR, backoff_factor),
        (MLFLOW_HTTP_REQUEST_BACKOFF_JITTER, 0),
    ]

    state: list[tuple] = []
    try:
        for env_var, new_value in swaps:
            state.append((env_var, *_swap(env_var, new_value)))
        yield
    finally:
        for env_var, was_set, previous in reversed(state):
            if was_set and previous is not None:
                env_var.set(previous)
            else:
                env_var.unset()
