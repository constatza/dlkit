"""Pure functions for process command-line inspection.

This module contains pure functions for inspecting process command lines
to determine if they are MLflow servers and if they match specific
host:port combinations. All functions are side-effect-free.
"""

from __future__ import annotations


def is_mlflow_process(cmdline: list[str]) -> bool:
    """Pure function to check if command line represents an MLflow server.

    Args:
        cmdline: Process command line arguments

    Returns:
        True if this appears to be an MLflow server process (requires uvicorn)
    """
    if not cmdline:
        return False

    cmdline_str = " ".join(cmdline)
    # MLflow server processes should surface uvicorn invocation or options
    has_uvicorn = "uvicorn" in cmdline_str or "--uvicorn-opts" in cmdline_str
    has_mlflow_server = "mlflow.server:app" in cmdline_str or "mlflow server" in cmdline_str

    return has_uvicorn and has_mlflow_server


def matches_host_port(cmdline: list[str], host: str, port: int) -> bool:
    """Pure function to check if command line matches host:port.

    Args:
        cmdline: Process command line arguments
        host: Target hostname
        port: Target port

    Returns:
        True if command line contains the host:port combination
    """
    if not cmdline:
        return False

    cmdline_str = " ".join(cmdline)
    host_variants = ["127.0.0.1", "localhost", "::1", host]

    if any(f"{variant}:{port}" in cmdline_str for variant in host_variants):
        return True

    has_host_flag = any(f"--host {variant}" in cmdline_str for variant in host_variants)
    has_port_flag = f"--port {port}" in cmdline_str

    return has_host_flag and has_port_flag
