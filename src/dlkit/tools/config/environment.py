"""Environment-aware settings using pydantic-settings for DLKit configuration.

This module provides centralized environment variable management following Single
Responsibility Principle. It manages only environment-level concerns like root_dir
and DLKit internal artifacts, while components maintain ownership of their specific paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dlkit.core.datatypes.secure_uris import SecurePath


def _configure_mlflow_retry_defaults() -> None:
    """Configure MLflow HTTP retry behavior with sensible defaults.

    MLflow defaults to 7 retries with exponential backoff, which can cause long waits
    when the server is unavailable. We reduce this to 2 retries for faster failure detection.
    Users can override by setting these environment variables before importing dlkit.

    Note: All MLflow HTTP environment variables must be integers (MLflow requirement).
    """
    if "MLFLOW_HTTP_REQUEST_MAX_RETRIES" not in os.environ:
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "2"
    if "MLFLOW_HTTP_REQUEST_TIMEOUT" not in os.environ:
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "5"
    if "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR" not in os.environ:
        os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"] = "1"


# Configure MLflow retry behavior early, before MLflow is imported
_configure_mlflow_retry_defaults()


class DLKitEnvironment(BaseSettings):
    """Environment-aware root configuration following SRP.

    This class manages only environment-level configuration that affects the entire
    DLKit system. Individual components maintain ownership of their specific paths.

    All settings can be overridden by environment variables prefixed with DLKIT_.
    For example, DLKIT_ROOT_DIR="/custom/root" will override the root_dir setting.
    """

    model_config = SettingsConfigDict(
        env_prefix="DLKIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment-level configuration (affects all path resolution)
    root_dir: SecurePath | None = Field(
        default=None,
        description="Root directory for relative path resolution across all components",
    )

    # DLKit internal artifacts only (not user dataflow)
    internal_dir: str = Field(
        default=".dlkit",
        description="Directory for DLKit internal artifacts (logs, server tracking)",
    )

    log_filename: str = Field(
        default="dlkit.log", description="Default log file name within internal directory"
    )

    server_tracking_file: str = Field(
        default="servers.json", description="Server tracking file name within internal directory"
    )

    def get_root_path(self) -> Path:
        """Get the effective root directory for path resolution.

        Returns:
            Path: Resolved root directory (uses current working directory if root_dir is None)
        """
        if self.root_dir is not None:
            # SecurePath already handles tilde expansion and security checks
            return Path(self.root_dir).resolve()
        return Path.cwd()

    def create_resolver_context(self) -> Any:
        """Create ResolverContext using existing factory infrastructure.

        Returns:
            ResolverContext: Context configured with effective root path
        """
        from dlkit.tools.io.resolution.factory import create_resolver_context

        return create_resolver_context(self.get_root_path())

    def get_internal_dir_path(self) -> Path:
        """Get path to DLKit internal directory.

        Returns:
            Path: Path to internal directory under root, creating if needed
        """
        internal_path = self.get_root_path() / self.internal_dir
        internal_path.mkdir(parents=True, exist_ok=True)
        return internal_path

    def get_log_file_path(self) -> Path:
        """Get path to default log file.

        Returns:
            Path: Path to log file in internal directory
        """
        return self.get_internal_dir_path() / self.log_filename

    def get_server_tracking_path(self) -> Path:
        """Get path to server tracking file.

        Returns:
            Path: Path to server tracking file in user's home .dlkit directory

        Note:
            Server tracking always goes in user's home directory for global access,
            not in the project's internal directory.
        """
        user_internal_dir = Path.home() / self.internal_dir
        user_internal_dir.mkdir(parents=True, exist_ok=True)
        return user_internal_dir / self.server_tracking_file


# Global instance for easy access throughout the system
env = DLKitEnvironment()
