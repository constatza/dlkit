"""Server configuration override application service.

This module provides functionality to apply CLI/API overrides to server
configurations without coupling to the source of the overrides.
"""

from __future__ import annotations

from typing import Any

from dlkit.tools.config.mlflow_settings import MLflowServerSettings
from dlkit.tools.utils.logging_config import get_logger

logger = get_logger(__name__)


class ServerConfigApplier:
    """Applies CLI/API overrides to server configuration.

    This service is responsible for taking override values from external
    sources (CLI arguments, API calls) and applying them to the server
    configuration in a type-safe manner.

    Responsibilities:
        - Extract valid override keys
        - Apply overrides using Pydantic model_copy
        - Preserve type safety through Pydantic validation
    """

    @staticmethod
    def apply_overrides(
        server_config: MLflowServerSettings, overrides: dict[str, Any]
    ) -> MLflowServerSettings:
        """Apply server configuration overrides.

        Args:
            server_config: Base server configuration
            overrides: Override values (typically from CLI or API)

        Returns:
            Updated server configuration with overrides applied
        """
        override_dict = {}
        for key in ["host", "port", "backend_store_uri", "artifacts_destination"]:
            if key in overrides:
                override_dict[key] = overrides[key]

        if override_dict:
            logger.debug("Applied server configuration overrides", overrides=override_dict)

        return server_config.model_copy(update=override_dict)
