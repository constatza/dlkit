"""Shared MLflow backend discovery helpers.

This module keeps backend selection and URI resolution acyclic.
"""

from __future__ import annotations

import socket
from urllib import request

import dlkit.tools.io.locations as locations

_LOCAL_MLFLOW_URL = "http://127.0.0.1:5000"


def local_host_alive() -> bool:
    """Check if a local MLflow tracking endpoint is reachable on 127.0.0.1:5000."""
    if not _tcp_port_open("127.0.0.1", 5000):
        return False

    return _looks_like_mlflow(_LOCAL_MLFLOW_URL)


def default_sqlite_backend_uri() -> str:
    """Return the configured local SQLite MLflow backend URI."""
    return locations.mlruns_backend_uri()


def _tcp_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError, AttributeError:
        return False


def _looks_like_mlflow(base_url: str) -> bool:
    """Confirm the service at base_url is MLflow by probing the /health endpoint."""
    try:
        url = f"{base_url}/health"
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False
