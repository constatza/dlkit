"""MLflow URI resolution for client-only tracking."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import dlkit.tools.io.locations as locations
from dlkit.tools.io import url_resolver

from .discovery import (
    default_sqlite_backend_uri as _default_sqlite_backend_uri,
)
from .discovery import (
    local_host_alive as _local_host_alive,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedMlflowUris:
    """Immutable MLflow endpoint resolution result."""

    tracking_uri: str
    artifact_uri: str | None
    scheme: str


def local_host_alive() -> bool:
    """Compatibility wrapper for local MLflow host probing."""
    return _local_host_alive()


def parse_mlflow_scheme(uri: str) -> str:
    """Parse and validate MLflow tracking URI scheme.

    Args:
        uri: MLflow tracking URI string.

    Returns:
        One of ``"http"``, ``"https"``, or ``"sqlite"``.

    Raises:
        ValueError: If the URI is empty or uses an unsupported scheme.
    """
    candidate = uri.strip()
    if not candidate:
        raise ValueError("MLflow URI cannot be empty")

    match candidate:
        case value if value.startswith("http://"):
            return "http"
        case value if value.startswith("https://"):
            return "https"
        case value if value.startswith("sqlite:///"):
            return "sqlite"
        case _:
            raise ValueError(
                f"Unsupported MLflow tracking URI scheme in '{uri}'. "
                "Supported schemes: http://, https://, sqlite:///"
            )


def resolve_tracking_uri() -> str:
    """Resolve MLflow tracking URI via backend selection.

    Delegates to :func:`~dlkit.runtime.tracking.backend.select_backend`
    for consistent backend selection logic.
    """
    from dlkit.runtime.tracking.backend import select_backend

    return select_backend(probe=local_host_alive).tracking_uri()


def resolve_artifact_uri(tracking_uri: str) -> str | None:
    """Resolve artifact URI for the resolved tracking URI."""
    env_artifact = os.getenv("MLFLOW_ARTIFACT_URI")
    if env_artifact:
        return env_artifact

    match parse_mlflow_scheme(tracking_uri):
        case "sqlite":
            return derive_sqlite_artifact_uri(tracking_uri)
        case "http" | "https":
            return None
        case unexpected:
            raise ValueError(f"Unsupported tracking URI scheme: {unexpected}")


def derive_sqlite_artifact_uri(tracking_uri: str) -> str:
    """Derive `<db_parent>/artifacts` for sqlite tracking URIs."""
    sqlite_db_path = _sqlite_db_path(tracking_uri)
    artifacts_dir = (sqlite_db_path.parent / "artifacts").resolve()
    return url_resolver.build_uri(artifacts_dir, scheme="file")


def resolve_mlflow_uris() -> ResolvedMlflowUris:
    """Resolve MLflow tracking and artifact URIs in one immutable result."""
    tracking_uri = resolve_tracking_uri()
    scheme = parse_mlflow_scheme(tracking_uri)
    artifact_uri = resolve_artifact_uri(tracking_uri)
    return ResolvedMlflowUris(
        tracking_uri=tracking_uri,
        artifact_uri=artifact_uri,
        scheme=scheme,
    )


def default_sqlite_backend_uri() -> str:
    """Compatibility wrapper for the configured local SQLite MLflow URI."""
    return locations.mlruns_backend_uri() or _default_sqlite_backend_uri()


def _normalize_tracking_uri(uri: str) -> str:
    """Normalize tracking URI and absolute-resolve sqlite paths."""
    cleaned = uri.strip()
    scheme = parse_mlflow_scheme(cleaned)

    match scheme:
        case "sqlite":
            return url_resolver.normalize_uri(cleaned, Path.cwd())
        case "http" | "https":
            return cleaned.rstrip("/")
        case unexpected:
            raise ValueError(f"Unsupported scheme for tracking URI normalization: {unexpected}")


def _sqlite_db_path(uri: str) -> Path:
    """Resolve sqlite URI path to an absolute DB path."""
    if not uri.startswith("sqlite:///"):
        raise ValueError(f"Expected sqlite URI, got '{uri}'")
    return url_resolver.resolve_local_uri(uri, Path.cwd())
