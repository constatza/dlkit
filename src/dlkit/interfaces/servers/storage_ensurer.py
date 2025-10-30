"""Server storage directory creation service.

This module provides functionality to ensure that required local directories
exist for MLflow server backend and artifacts storage.
"""

from __future__ import annotations

from urllib.parse import urlparse

from dlkit.tools.config.mlflow_settings import MLflowServerSettings
from dlkit.tools.utils.system_utils import mkdir_for_local
from dlkit.tools.utils.logging_config import get_logger

logger = get_logger(__name__)


class ServerStorageEnsurer:
    """Ensures local storage directories exist for server operation.

    This service is responsible for creating the necessary local directories
    for MLflow backend store and artifacts storage when they are configured
    as local file paths.

    Responsibilities:
        - Detect local file-based storage configurations
        - Create required directories with proper error handling
        - Skip remote storage configurations
        - Handle both file:// and sqlite:// URIs
    """

    def ensure_storage(self, config: MLflowServerSettings) -> None:
        """Create local storage directories for backend & artifacts if needed.

        Args:
            config: MLflow server configuration

        Raises:
            RuntimeError: If directory creation fails
        """
        logger.debug("Ensuring local storage directories")
        local_hosts = {"localhost", "127.0.0.1", "::1", None}

        for attr in ("backend_store_uri", "artifacts_destination"):
            uri = getattr(config, attr)
            logger.debug(f"Checking {attr} = {uri}")
            if uri is None:
                continue

            try:
                parsed = urlparse(str(uri))
            except Exception:
                parsed = None

            is_file = bool(parsed and (parsed.scheme in ("", "file")))
            host_local = bool(parsed and (parsed.hostname in local_hosts))
            logger.debug(f"{attr} is_file={is_file}, host_local={host_local}")

            if is_file or host_local:
                try:
                    # mkdir_for_local now handles Pydantic Url objects and sqlite:// URIs directly
                    # Just pass the URI as-is
                    uri_str = str(uri)
                    logger.debug(f"Creating directory for {attr}: {uri_str}")
                    mkdir_for_local(uri_str)
                    logger.debug(f"Successfully created directory for {attr}")
                except Exception as e:
                    logger.error(f"Failed to create local directory for {attr}: {e}")
                    raise RuntimeError(f"Directory creation failed for {attr}: {e}") from e

        logger.debug("Local storage directories ensured successfully")
