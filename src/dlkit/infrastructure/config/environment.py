"""Environment-aware settings using pydantic-settings for DLKit configuration."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    """

    model_config = SettingsConfigDict(
        env_prefix="DLKIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
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
            Path: Current working directory
        """
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

ensure_mlflow_defaults()
