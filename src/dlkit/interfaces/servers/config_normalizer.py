"""Server configuration path normalization service.

This module provides path normalization for MLflow server configurations,
ensuring that relative paths are resolved relative to the environment root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from dlkit.core.datatypes.urls import ArtifactDestination
from dlkit.tools.config.mlflow_settings import MLflowServerSettings
from dlkit.tools.io import locations, url_resolver
from dlkit.tools.io.path_normalizers import normalize_file_uri, resolve_local_path
from dlkit.tools.utils.logging_config import get_logger

logger = get_logger(__name__)

_ARTIFACT_ADAPTER = TypeAdapter(ArtifactDestination)


class ServerConfigNormalizer:
    """Normalizes server configuration paths relative to environment root."""

    def normalize(self, config: MLflowServerSettings) -> MLflowServerSettings:
        """Normalize server paths relative to current root context."""

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
        """Normalize backend store URI relative to root directory."""

        # url_resolver handles both sqlite:// and file:// URIs
        if url_resolver.is_local_uri(uri) or not ("://" in uri):
            try:
                return url_resolver.normalize_uri(uri, root_dir)
            except ValueError:
                # If normalization fails, pass through for remote URIs
                pass

        # Non-local URIs (http://, s3://, etc.) pass through unchanged
        return uri

    @staticmethod
    def _normalize_artifacts_destination(destination: str, root_dir: Path) -> str:
        """Normalize artifacts destination relative to root directory."""

        normalized = _ARTIFACT_ADAPTER.validate_python(destination)

        normalized_file = normalize_file_uri(normalized, root_dir)
        if normalized_file:
            return normalized_file

        if "://" in normalized:
            return normalized

        resolved = resolve_local_path(normalized, root_dir)
        return resolved.as_posix()
