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

from dlkit.infrastructure.config.security.uri_types import SecurePath


def _setenv_if_missing(key: str, value: int) -> None:
    """Set environment variable only if not already present.

    Args:
        key: Environment variable name.
        value: Integer value (MLflow reads these as strings internally).
    """
    if key not in os.environ:
        os.environ[key] = str(value)


_mlflow_defaults_configured = False


def ensure_mlflow_defaults() -> None:
    """Configure MLflow HTTP retry behavior with sensible defaults (lazy, idempotent).

    MLflow defaults to 7 retries with exponential backoff, which can cause long waits
    when the server is unavailable. We reduce this to 2 retries for faster failure detection.
    Users can override by setting these environment variables before this is called.

    This function is idempotent and safe to call multiple times. It runs lazily on first
    use rather than at import time to avoid hidden import-time side effects.

    Note: All MLflow HTTP environment variables must be integers (MLflow requirement).
    """
    global _mlflow_defaults_configured
    if _mlflow_defaults_configured:
        return
    _setenv_if_missing("MLFLOW_HTTP_REQUEST_MAX_RETRIES", 2)
    _setenv_if_missing("MLFLOW_HTTP_REQUEST_TIMEOUT", 5)
    _setenv_if_missing("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", 1)
    _mlflow_defaults_configured = True


class EnvironmentSettings(BaseSettings):
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
        description="Directory for DLKit internal artifacts (logs)",
    )

    log_filename: str = Field(
        default="dlkit.log", description="Default log file name within internal directory"
    )

    log_level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Set via DLKIT_LOG_LEVEL env var.",
    )

    log_file: str | None = Field(
        default=None,
        description="Custom log file path (overrides default). Set via DLKIT_LOG_FILE env var.",
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


# Global instance for easy access throughout the system
env = EnvironmentSettings()

# Configure MLflow retry behavior early, before MLflow is imported.
# ensure_mlflow_defaults() is idempotent so this is safe to call at module level.
ensure_mlflow_defaults()


def sync_session_root_to_environment(settings: Any) -> None:
    """Synchronize SESSION.root_dir to the global environment fallback.

    This is intentionally config-local so `tools.config` does not depend on
    `tools.io` for root propagation.
    """

    try:
        from loguru import logger

        if os.environ.get("DLKIT_ROOT_DIR"):
            return

        session = getattr(settings, "SESSION", None)
        if session is None:
            return

        session_root_dir = getattr(session, "root_dir", None)
        if session_root_dir is None:
            return

        root_path = Path(session_root_dir).expanduser()
        if not root_path.is_absolute():
            root_path = (Path.cwd() / root_path).resolve()
        else:
            root_path = root_path.resolve()

        env.root_dir = str(root_path)
        logger.debug(
            "Synchronized SESSION.root_dir to EnvironmentSettings for fallback path resolution",
            session_root_dir=str(root_path),
        )
    except Exception as e:
        from loguru import logger

        logger.warning(f"Failed to sync SESSION.root_dir to environment (non-fatal): {e}")
