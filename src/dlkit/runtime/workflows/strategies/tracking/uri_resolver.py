"""MLflow URI resolution for client-only tracking."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib import request

import dlkit.tools.io.locations as locations
from dlkit.tools.io import url_resolver

_LOCAL_MLFLOW_URL = "http://127.0.0.1:5000"


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedMlflowUris:
    """Immutable MLflow endpoint resolution result."""

    tracking_uri: str
    artifact_uri: str | None
    scheme: str


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


def local_host_alive() -> bool:
    """Check if a local MLflow tracking endpoint is reachable on 127.0.0.1:5000."""
    if not _tcp_port_open("127.0.0.1", 5000):
        return False

    return _looks_like_mlflow(_LOCAL_MLFLOW_URL)


def resolve_tracking_uri() -> str:
    """Resolve MLflow tracking URI via backend selection.

    Delegates to :func:`~dlkit.runtime.workflows.strategies.tracking.backend.select_backend`
    for consistent backend selection logic.
    """
    from dlkit.runtime.workflows.strategies.tracking.backend import select_backend

    return select_backend().tracking_uri()


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
    """Return the configured local SQLite MLflow backend URI."""
    return locations.mlruns_backend_uri()


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


def _tcp_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError, AttributeError:
        return False


def _looks_like_mlflow(base_url: str) -> bool:
    """Confirm the service at base_url is MLflow by probing the /health endpoint.

    Uses ``/health`` (present in MLflow 2.x and 3.x), which returns HTTP 200 with
    body ``"OK"``.  The old ``/api/2.0/mlflow/experiments/list`` was removed in
    MLflow 3.x.

    Args:
        base_url: Base URL of the service to probe (e.g. ``"http://127.0.0.1:5000"``).

    Returns:
        True only when the MLflow health endpoint returns HTTP 200.
    """
    try:
        url = f"{base_url}/health"
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False
