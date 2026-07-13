"""Environment-aware settings using pydantic-settings for DLKit configuration."""

from __future__ import annotations

import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
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


def set_mlflow_max_retries(max_retries: int) -> None:
    """Explicitly override MLflow's HTTP retry budget for this process.

    Unlike ``_setenv_if_missing``, this always overwrites the env var. MLflow
    re-reads it on every HTTP call, so this takes effect immediately for any
    request made after it runs, regardless of import order.

    Args:
        max_retries: Maximum number of retries before an MLflow HTTP request
            gives up (MLflow reads this as a string internally).
    """
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = str(max_retries)


_mlflow_defaults_configured = False


def ensure_mlflow_defaults() -> None:
    """Configure MLflow HTTP retry behavior with sensible defaults (idempotent).

    MLflow's stock defaults (7 retries, 120s timeout, backoff factor 2) can cause
    multi-minute hangs when a server is truly unreachable. We tighten them, but not
    so far that a lightly-loaded local ``mlflow server`` can't absorb a short burst
    of transient 5xx responses under concurrent load (e.g. several optimization runs
    hitting the same local server at once) — 2 retries / 5s timeout proved too tight
    in practice and turned recoverable contention into hard failures. Users can
    override by setting these environment variables before this is called, or via
    ``TrackingSettings.max_retries`` (see ``set_mlflow_max_retries``).

    This function is idempotent and safe to call multiple times. It currently runs
    at import time (see bottom of this module).

    Note: All MLflow HTTP environment variables must be integers (MLflow requirement).
    """
    global _mlflow_defaults_configured
    if _mlflow_defaults_configured:
        return
    _setenv_if_missing("MLFLOW_HTTP_REQUEST_MAX_RETRIES", 5)
    _setenv_if_missing("MLFLOW_HTTP_REQUEST_TIMEOUT", 30)
    _setenv_if_missing("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", 2)
    _mlflow_defaults_configured = True


_BEST_EFFORT_RETRY_BUDGET: dict[str, str] = {
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "2",
    "MLFLOW_HTTP_REQUEST_TIMEOUT": "5",
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": "1",
}
_best_effort_retry_budget_lock = threading.RLock()


@contextmanager
def best_effort_retry_budget() -> Generator[None]:
    """Temporarily tighten MLflow's HTTP retry env vars for one best-effort call.

    ``ensure_mlflow_defaults``/``set_mlflow_max_retries`` size the retry budget
    for MLflow calls that must not silently fail (e.g. ``log_model`` for the
    best-retrain run). Calls wrapped in ``best_effort`` are allowed to fail —
    there's no reason for them to burn that same budget waiting out a server
    that's already returning errors. This scopes them back down to the
    proven-in-practice pre-``bf66c9b`` values (2 retries / 5s timeout /
    backoff factor 1) for the duration of the call, then restores whatever
    value (or absence) was present beforehand — so a
    ``TrackingSettings.max_retries`` override survives around it.

    Uses an ``RLock`` because ``best_effort``-wrapped calls invoke other
    ``best_effort``-wrapped calls on the same thread (e.g.
    ``log_best_trial_result`` -> ``log_trial_artifacts`` in
    ``optimization/infrastructure/tracking.py``); a plain ``Lock`` would
    deadlock on the first nested call.
    """
    with _best_effort_retry_budget_lock:
        previous = {name: os.environ.get(name) for name in _BEST_EFFORT_RETRY_BUDGET}
        os.environ.update(_BEST_EFFORT_RETRY_BUDGET)
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


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
        description="Directory for DLKit internal artifacts (logs, local MLflow tracking DB)",
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


# Global instance for easy access throughout the system
env = EnvironmentSettings()

ensure_mlflow_defaults()
