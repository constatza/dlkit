"""MLflow client factory for proper client instance management."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import mlflow
from mlflow import MlflowClient

from dlkit.tools.config.mlflow_settings import MLflowClientSettings
from dlkit.tools.io.url_utils import parse_url
from mlflow.environment_variables import (
    MLFLOW_HTTP_REQUEST_TIMEOUT,
    MLFLOW_HTTP_REQUEST_MAX_RETRIES,
    MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR,
    MLFLOW_HTTP_REQUEST_BACKOFF_JITTER,
)
from dlkit.tools.utils.logging_config import get_logger

logger = get_logger(__name__)


class MLflowClientFactory:
    """Factory for creating properly configured MLflow client instances.

    This factory eliminates dependency on global MLflow state by creating
    client instances with explicit tracking URI configuration.
    """

    @staticmethod
    def create_client(
        client_config: MLflowClientSettings | None = None,
        tracking_uri: str | None = None,
    ) -> MlflowClient:
        """Create MLflow client with explicit configuration.

        Args:
            client_config: Client configuration settings
            tracking_uri: Override tracking URI

        Returns:
            Configured MlflowClient instance
        """
        # Determine tracking URI from various sources
        uri = None

        if tracking_uri:
            uri = str(tracking_uri)
        elif client_config and hasattr(client_config, "tracking_uri"):
            uri = str(client_config.tracking_uri)

        # Validate tracking URI if provided
        if uri:
            try:
                parsed = parse_url(uri)
                if not parsed.scheme:
                    logger.warning(f"Tracking URI missing scheme: {uri}")
            except Exception as e:
                logger.warning(f"Invalid tracking URI format: {uri} - {e}")

        # Create client with explicit tracking URI
        if uri:
            logger.debug(f"Creating MLflow client with tracking URI: {uri}")
            return MlflowClient(tracking_uri=uri)
        else:
            logger.debug("Creating MLflow client with default tracking URI")
            return MlflowClient()

    @staticmethod
    def create_client_from_server_info(
        server_info: Any,
        client_config: MLflowClientSettings | None = None,
    ) -> MlflowClient:
        """Create MLflow client configured for specific server.

        Args:
            server_info: Server information containing URL
            client_config: Additional client configuration

        Returns:
            MlflowClient configured for the server
        """
        server_url = getattr(server_info, "url", None)
        if not server_url:
            logger.warning("Server info missing URL, using default client")
            return MLflowClientFactory.create_client(client_config)

        logger.debug(f"Creating MLflow client for server: {server_url}")
        return MLflowClientFactory.create_client(
            client_config=client_config,
            tracking_uri=server_url,
        )

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
            logger.warning(f"MLflow client connectivity test failed: {e}")
            return False

    @staticmethod
    def get_or_create_experiment(
        client: MlflowClient,
        experiment_name: str,
    ) -> str:
        """Get experiment ID or create if it doesn't exist."""

        logger.debug(f"Checking if experiment '{experiment_name}' exists")
        try:
            logger.debug(f"Calling client.get_experiment_by_name('{experiment_name}')")
            experiment = client.get_experiment_by_name(experiment_name)
            logger.debug(f"get_experiment_by_name returned: {experiment}")
            if experiment is not None:
                logger.debug(f"Experiment exists with ID: {experiment.experiment_id}")
                return experiment.experiment_id
        except Exception as e:
            logger.debug(f"get_experiment_by_name failed with: {type(e).__name__}: {e}")
            pass

        # Create new experiment if not found
        logger.debug(f"Creating new MLflow experiment: {experiment_name}")
        result = client.create_experiment(experiment_name)
        logger.debug(f"Experiment created with ID: {result}")
        return result


@contextmanager
def _temporary_request_settings(
    timeout: int,
    max_retries: int,
    backoff_factor: int,
) -> None:
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
