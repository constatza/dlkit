"""Sealed sum type for MLflow tracking backends.

Replaces stringly-typed scheme checks with typed backend objects.
All scheme-specific logic lives on backend instances.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from dlkit.infrastructure.io import url_resolver

from .discovery import default_sqlite_backend_uri, local_host_alive


@runtime_checkable
class ITrackingBackend(Protocol):
    """Protocol for MLflow tracking backends."""

    def tracking_uri(self) -> str: ...
    def artifact_uri(self) -> str | None: ...
    def scheme(self) -> str: ...


@dataclass(frozen=True, slots=True)
class RemoteServerBackend:
    """Explicit HTTP/HTTPS server.

    Attributes:
        uri: Normalized HTTP/HTTPS tracking URI.
    """

    uri: str

    def tracking_uri(self) -> str:
        """Return the configured remote tracking URI."""
        return self.uri

    def artifact_uri(self) -> str | None:
        """Return None — remote server manages artifact storage."""
        return None

    def scheme(self) -> str:
        """Return 'https' or 'http' based on URI prefix."""
        return "https" if self.uri.startswith("https") else "http"


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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
        """Return artifact URI derived from db_path parent."""
        return url_resolver.build_uri(self.db_path.parent / "artifacts", scheme="file")

    def scheme(self) -> str:
        """Return 'sqlite'."""
        return "sqlite"


TrackingBackend = RemoteServerBackend | LocalServerBackend | LocalSqliteBackend


def select_backend(
    *,
    uri: str | None = None,
    probe: Callable[[], bool] | None = None,
) -> TrackingBackend:
    """Select the appropriate tracking backend.

    Selection logic:
    1. Explicit ``uri`` from config → typed backend (http/https or sqlite)
    2. Local server probe succeeds → ``LocalServerBackend``
    3. Fallback → ``LocalSqliteBackend`` derived from locations

    Args:
        uri: Tracking URI from ``TrackingSettings.uri``. When set, takes
            precedence over probe and fallback; env vars are never consulted.
        probe: Callable returning True if a local MLflow server is reachable.
            Defaults to :func:`~dlkit.engine.tracking.discovery.local_host_alive`.

    Returns:
        The selected ``TrackingBackend`` instance.

    Raises:
        ValueError: If ``uri`` is set but uses an unsupported scheme.
    """
    if probe is None:
        probe = local_host_alive

    if uri:
        cleaned = uri.strip()
        if cleaned.startswith("http://") or cleaned.startswith("https://"):
            return RemoteServerBackend(uri=cleaned.rstrip("/"))
        if cleaned.startswith("sqlite:///"):
            db_path = url_resolver.resolve_local_uri(cleaned, Path.cwd())
            return LocalSqliteBackend(db_path=db_path)
        raise ValueError(
            f"Unsupported MLflow tracking URI scheme in '{uri}'. "
            "Supported: http://, https://, sqlite:///"
        )

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
