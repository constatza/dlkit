"""MLflow URI resolution for client-only tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib import request
import os
import socket

from dlkit.tools.io import locations


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
    """Resolve MLflow tracking URI with env -> localhost probe -> sqlite fallback."""
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return _normalize_tracking_uri(env_uri)

    if local_host_alive():
        return _LOCAL_MLFLOW_URL

    return _normalize_tracking_uri(locations.mlruns_backend_uri())


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
    return artifacts_dir.as_uri()


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


def _normalize_tracking_uri(uri: str) -> str:
    """Normalize tracking URI and absolute-resolve sqlite paths."""
    cleaned = uri.strip()
    scheme = parse_mlflow_scheme(cleaned)

    match scheme:
        case "sqlite":
            db_path = _sqlite_db_path(cleaned)
            return f"sqlite:///{db_path.as_posix()}"
        case "http" | "https":
            return cleaned.rstrip("/")
        case unexpected:
            raise ValueError(f"Unsupported scheme for tracking URI normalization: {unexpected}")


def _sqlite_db_path(uri: str) -> Path:
    """Resolve sqlite URI path to an absolute DB path."""
    if not uri.startswith("sqlite:///"):
        raise ValueError(f"Expected sqlite URI, got '{uri}'")

    raw_path = uri[len("sqlite:///") :]
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _tcp_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except (OSError, AttributeError):
        return False


def _looks_like_mlflow(base_url: str) -> bool:
    """Confirm the service at base_url is MLflow by probing a canonical REST endpoint.

    Uses the unambiguous ``/api/2.0/mlflow/experiments/list`` path instead of generic
    heuristics so that any non-MLflow service on port 5000 is not mistakenly accepted.

    Args:
        base_url: Base URL of the service to probe (e.g. ``"http://127.0.0.1:5000"``).

    Returns:
        True only when the canonical MLflow experiments endpoint returns HTTP 200.
    """
    try:
        url = f"{base_url}/api/2.0/mlflow/experiments/list"
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=1.0) as resp:  # noqa: S310 - controlled localhost probe
            return resp.status == 200
    except Exception:
        return False
