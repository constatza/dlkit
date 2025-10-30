"""Server configuration path normalization service.

This module provides path normalization for MLflow server configurations,
ensuring that relative paths are resolved relative to the environment root.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from dlkit.tools.config.mlflow_settings import MLflowServerSettings
from dlkit.tools.io import locations
from dlkit.tools.utils.logging_config import get_logger

logger = get_logger(__name__)


class ServerConfigNormalizer:
    """Normalizes server configuration paths relative to environment root.

    This service is responsible for ensuring that relative paths in the
    server configuration are resolved to absolute paths relative to the
    DLKit environment root directory.

    Responsibilities:
        - Resolve relative backend_store_uri paths
        - Resolve relative artifacts_destination paths
        - Preserve absolute paths and remote URIs
        - Handle various URI schemes (file://, sqlite://, etc.)
    """

    def normalize(self, config: MLflowServerSettings) -> MLflowServerSettings:
        """Normalize server paths relative to current root context.

        Args:
            config: Server configuration to normalize

        Returns:
            Normalized server configuration with absolute paths
        """
        try:
            root_dir = locations.root()
        except Exception:
            root_dir = Path.cwd()

        updates: dict[str, Any] = {}

        backend_uri = getattr(config, "backend_store_uri", None)
        if backend_uri:
            backend_str = str(backend_uri)
            normalized_backend = self._normalize_backend_uri(backend_str, root_dir)
            if normalized_backend != backend_str:
                updates["backend_store_uri"] = normalized_backend

        artifacts_dest = getattr(config, "artifacts_destination", None)
        if artifacts_dest:
            artifacts_str = str(artifacts_dest)
            normalized_artifacts = self._normalize_artifacts_destination(artifacts_str, root_dir)
            if normalized_artifacts != artifacts_str:
                updates["artifacts_destination"] = normalized_artifacts

        if updates:
            return config.model_copy(update=updates)
        return config

    @staticmethod
    def _normalize_backend_uri(uri: str, root_dir: Path) -> str:
        """Normalize backend store URI relative to root directory.

        Args:
            uri: Backend store URI to normalize
            root_dir: Root directory for path resolution

        Returns:
            Normalized backend store URI
        """
        parsed = urlparse(uri)
        if parsed.scheme not in {"file", "sqlite"}:
            return uri

        path = Path(parsed.path)
        if path.is_absolute():
            return uri

        resolved = (root_dir / path).resolve()
        if parsed.scheme == "sqlite":
            return f"sqlite:///{resolved.as_posix()}"
        return resolved.as_uri()

    @staticmethod
    def _normalize_artifacts_destination(destination: str, root_dir: Path) -> str:
        """Normalize artifacts destination relative to root directory.

        Args:
            destination: Artifacts destination to normalize
            root_dir: Root directory for path resolution

        Returns:
            Normalized artifacts destination
        """
        parsed = urlparse(destination)

        if parsed.scheme in {"", None}:
            path = Path(destination)
            if path.is_absolute():
                return destination
            return str((root_dir / path).resolve())

        if parsed.scheme == "file":
            path = Path(parsed.path)
            if path.is_absolute():
                return destination
            resolved = (root_dir / path).resolve()
            return resolved.as_uri()

        return destination
