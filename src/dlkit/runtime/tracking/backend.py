"""Sealed sum type for MLflow tracking backends.

Replaces stringly-typed scheme checks with typed backend objects.
All scheme-specific logic lives on backend instances.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dlkit.tools.io import url_resolver

from .discovery import default_sqlite_backend_uri, local_host_alive


class ITrackingBackend(Protocol):
    """Protocol for MLflow tracking backends."""

    def tracking_uri(self) -> str: ...
    def artifact_uri(self) -> str | None: ...
    def scheme(self) -> str: ...


@dataclass(frozen=True)
class RemoteServerBackend:
    """Explicit HTTP/HTTPS server configured by the user via env var.

    Attributes:
        uri: Normalized HTTP/HTTPS tracking URI.
    """

    uri: str

    def tracking_uri(self) -> str:
        """Return the configured remote tracking URI."""
        return self.uri

    def artifact_uri(self) -> str | None:
        """Return artifact URI from env or None."""
        return os.getenv("MLFLOW_ARTIFACT_URI")

    def scheme(self) -> str:
        """Return 'https' or 'http' based on URI prefix."""
        return "https" if self.uri.startswith("https") else "http"


@dataclass(frozen=True)
class LocalServerBackend:
    """Auto-detected local MLflow server on 127.0.0.1:5000."""

    def tracking_uri(self) -> str:
        """Return the local server tracking URI."""
        return "http://127.0.0.1:5000"

    def artifact_uri(self) -> str | None:
        """Return None (server-side artifact storage)."""
        return None

    def scheme(self) -> str:
        """Return 'http'."""
        return "http"


@dataclass(frozen=True)
class LocalSqliteBackend:
    """Local SQLite backend.

    Attributes:
        db_path: Absolute path to the SQLite database file.
    """

    db_path: Path

    def tracking_uri(self) -> str:
        """Return sqlite:/// URI for the database path."""
        return url_resolver.build_uri(self.db_path, scheme="sqlite")

    def artifact_uri(self) -> str | None:
        """Return artifact URI from env var or derive from db_path parent."""
        env = os.getenv("MLFLOW_ARTIFACT_URI")
        if env:
            return env
        return url_resolver.build_uri(self.db_path.parent / "artifacts", scheme="file")

    def scheme(self) -> str:
        """Return 'sqlite'."""
        return "sqlite"


TrackingBackend = RemoteServerBackend | LocalServerBackend | LocalSqliteBackend


def select_backend(
    *,
    probe: Callable[[], bool] | None = None,
) -> TrackingBackend:
    """Select the appropriate tracking backend.

    Selection logic:
    1. ``MLFLOW_TRACKING_URI`` env var (HTTP/HTTPS or ``sqlite:///``) → typed backend
    2. Local server probe succeeds → ``LocalServerBackend``
    3. Fallback → ``LocalSqliteBackend`` derived from locations

    Note:
        SQLite URIs in ``MLFLOW_TRACKING_URI`` are honoured explicitly here
        because ``mlflow.set_tracking_uri(None)`` in MLflow 3.x resets the
        internal state to the CWD-relative default (``sqlite:///mlflow.db``)
        rather than re-reading the env var.  Without this branch, the env-var
        isolation set up by test fixtures would be silently ignored and every
        ``reset_global_state()`` call could leak a ``mlflow.db`` into the
        project root.

    Args:
        probe: Callable returning True if a local MLflow server is reachable.
            Defaults to :func:`~dlkit.runtime.tracking.discovery.local_host_alive`.

    Returns:
        The selected ``TrackingBackend`` instance.
    """
    if probe is None:
        probe = local_host_alive

    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        if env_uri.startswith("http://") or env_uri.startswith("https://"):
            return RemoteServerBackend(uri=env_uri.strip().rstrip("/"))
        if env_uri.startswith("sqlite:///"):
            db_path = url_resolver.resolve_local_uri(env_uri, Path.cwd())
            return LocalSqliteBackend(db_path=db_path)

    if probe():
        return LocalServerBackend()

    db_path = _resolve_sqlite_db_path()
    return LocalSqliteBackend(db_path=db_path)


def _resolve_sqlite_db_path() -> Path:
    """Resolve the SQLite database path from locations.

    Returns:
        Absolute path to the SQLite database file.
    """
    raw = default_sqlite_backend_uri()
    return url_resolver.resolve_local_uri(raw, Path.cwd())
