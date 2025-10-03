"""Pure domain functions following functional programming principles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dlkit.tools.config.environment import DLKitEnvironment
from dlkit.tools.io import locations


# Global environment instance for domain functions
env = DLKitEnvironment()


def validate_mlflow_config(mlflow_cfg: Any) -> None:
    """Pure function to validate MLflow configuration.

    Args:
        mlflow_cfg: MLflow configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    if mlflow_cfg is None:
        raise ValueError("MLFLOW configuration is missing")

    if not getattr(mlflow_cfg, "enabled", False):
        raise ValueError("MLflow is not enabled")


def get_host_variants(host: str, port: int) -> list[str]:
    """Pure function to get all equivalent host:port combinations.

    Args:
        host: Server hostname
        port: Server port

    Returns:
        List of equivalent host:port strings
    """
    variants = [f"{host}:{port}"]

    # Add localhost variants for local addresses
    if host == "localhost":
        variants.append(f"127.0.0.1:{port}")
    elif host == "127.0.0.1":
        variants.append(f"localhost:{port}")
    elif host not in ("127.0.0.1", "localhost"):
        variants.extend([f"127.0.0.1:{port}", f"localhost:{port}"])

    # Remove duplicates while preserving order
    return list(dict.fromkeys(variants))


def get_tracking_file_path() -> Path:
    """Get path to server tracking file under the user's home directory.

    Returns:
        Path: ``~/.dlkit/servers.json``
    """
    return env.get_server_tracking_path()


def load_tracking_data(tracking_file: Path) -> dict[str, list[int]]:
    """Pure function to load tracking dataflow from file.

    Args:
        tracking_file: Path to tracking file

    Returns:
        Dictionary mapping server keys to PID lists
    """
    if not tracking_file.exists():
        return {}

    try:
        with tracking_file.open("r") as f:
            data = json.load(f)

        # Ensure all values are lists of integers
        return {
            k: [int(pid) for pid in v if isinstance(pid, (int, str))]
            for k, v in data.items()
            if isinstance(v, list)
        }
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return {}


def save_tracking_data(tracking_file: Path, data: dict[str, list[int]]) -> None:
    """Pure function to save tracking dataflow to file.

    Args:
        tracking_file: Path to tracking file
        data: Dictionary mapping server keys to PID lists

    Raises:
        OSError: If file cannot be written
    """
    # Ensure parent directory exists
    tracking_file.parent.mkdir(parents=True, exist_ok=True)

    with tracking_file.open("w") as f:
        json.dump(data, f, indent=2)


def should_use_default_storage(server_config: Any, overrides: dict[str, Any]) -> bool:
    """Pure function to determine if default storage setup is needed.

    Args:
        server_config: Server configuration object (could be MLflowServerSettings or MLflowServerContext)
        overrides: CLI parameter overrides

    Returns:
        True if default storage setup is needed
    """
    # If user provided explicit storage options, respect them
    if (
        overrides.get("backend_store_uri") is not None
        or overrides.get("artifacts_destination") is not None
    ):
        return False

    # Get the actual settings object - could be MLflowServerContext wrapping settings
    # or direct MLflowServerSettings
    actual_config = server_config
    if (
        hasattr(server_config, "_server_config")
        and server_config.__class__.__module__ == "dlkit.interfaces.servers.mlflow_adapter"
    ):
        # This is a MLflowServerContext, get the wrapped settings
        actual_config = server_config._server_config

    # If config already has storage configured, use it
    backend_uri = getattr(actual_config, "backend_store_uri", None)
    artifacts_dest = getattr(actual_config, "artifacts_destination", None)

    # Check if either setting is configured (not None and not empty string)
    if (
        backend_uri is not None
        and str(backend_uri).strip() != ""
        or artifacts_dest is not None
        and str(artifacts_dest).strip() != ""
    ):
        return False

    return True


def resolve_component_path(path_value: str | Path | None) -> Path | None:
    """Resolve component paths using existing resolver infrastructure.

    This function leverages the existing path resolution system to handle
    relative/absolute paths, tilde expansion, and security checks. It also
    respects API path overrides when available.

    Args:
        path_value: Path to resolve (can be relative or absolute)

    Returns:
        Resolved absolute path, or None if path_value is None
    """
    if not path_value:
        return None

    # Always try the context-aware resolution first
    try:
        from dlkit.interfaces.api.overrides.path_context import resolve_with_context

        return resolve_with_context(str(path_value), env)
    except ImportError:
        # Fallback to direct resolution if path_context is not available
        from dlkit.tools.io.resolution.factory import create_default_resolver_system

        registry, context = create_default_resolver_system(env.get_root_path())
        resolved = registry.resolve(path_value, context)

        # Ensure we return a Path object
        return Path(resolved) if resolved is not None else None


def get_default_output_dir() -> Path:
    """Get default output directory with environment-aware root resolution.

    Returns:
        Path to default output directory under environment root
    """
    # Use centralized locations policy
    return locations.output()


def get_default_mlruns_path() -> Path:
    """Get default MLruns path with environment-aware root resolution.

    Returns:
        Path to default MLruns directory under output/
    """
    # Backward-compatible default under output/
    return locations.output("mlruns")


def get_default_optuna_storage_url() -> str:
    """Get default Optuna storage URL with environment-aware root resolution.

    Returns:
        Default Optuna storage URL under output/
    """
    # Centralized default storage URI
    return locations.optuna_storage_uri()


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


def add_server_to_tracking(
    servers: dict[str, list[int]], host: str, port: int, pid: int
) -> dict[str, list[int]]:
    """Pure function to add server to tracking

    Args:
        servers: Current tracking dataflow
        host: Server hostname
        port: Server port
        pid: Process ID

    Returns:
        Updated tracking dataflow
    """
    servers_copy = servers.copy()
    server_key = f"{host}:{port}"

    if server_key not in servers_copy:
        servers_copy[server_key] = []

    if pid not in servers_copy[server_key]:
        servers_copy[server_key] = servers_copy[server_key] + [pid]

    return servers_copy


def remove_server_from_tracking(
    servers: dict[str, list[int]], host: str, port: int
) -> dict[str, list[int]]:
    """Pure function to remove server from tracking

    Args:
        servers: Current tracking dataflow
        host: Server hostname
        port: Server port

    Returns:
        Updated tracking dataflow
    """
    servers_copy = servers.copy()
    server_keys_to_remove = get_host_variants(host, port)

    for key in server_keys_to_remove:
        servers_copy.pop(key, None)

    return servers_copy


def get_pids_for_server(servers: dict[str, list[int]], host: str, port: int) -> list[int]:
    """Pure function to get PIDs for a server from tracking

    Args:
        servers: Tracking dataflow
        host: Server hostname
        port: Server port

    Returns:
        List of tracked PIDs for the server
    """
    server_keys = get_host_variants(host, port)

    pids = []
    for key in server_keys:
        if key in servers:
            pids.extend(servers[key])

    return pids
