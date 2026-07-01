"""MLflow URI utilities."""

from __future__ import annotations

from .discovery import (
    local_host_alive as _local_host_alive,
)


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
