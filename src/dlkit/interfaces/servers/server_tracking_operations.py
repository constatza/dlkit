"""Pure functions for server tracking data manipulation.

This module contains pure, immutable functions for managing server
tracking data. All functions are side-effect-free, taking tracking
data as input and returning new tracking data as output.

These functions follow functional programming principles:
- Pure functions (no side effects)
- Immutable data structures (copy on write)
- Referential transparency
"""

from __future__ import annotations

import json
from pathlib import Path


def load_tracking_data(tracking_file: Path) -> dict[str, list[int]]:
    """Pure function to load tracking data from file.

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
    """Pure function to save tracking data to file.

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


def add_server_to_tracking(
    servers: dict[str, list[int]], host: str, port: int, pid: int
) -> dict[str, list[int]]:
    """Pure function to add server to tracking data.

    Args:
        servers: Current tracking data
        host: Server hostname
        port: Server port
        pid: Process ID

    Returns:
        Updated tracking data (new dict, original unchanged)
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
    """Pure function to remove server from tracking data.

    Args:
        servers: Current tracking data
        host: Server hostname
        port: Server port

    Returns:
        Updated tracking data (new dict, original unchanged)
    """
    from .server_configuration import get_host_variants

    servers_copy = servers.copy()
    server_keys_to_remove = get_host_variants(host, port)

    for key in server_keys_to_remove:
        servers_copy.pop(key, None)

    return servers_copy


def get_pids_for_server(servers: dict[str, list[int]], host: str, port: int) -> list[int]:
    """Pure function to get PIDs for a server from tracking data.

    Args:
        servers: Tracking data
        host: Server hostname
        port: Server port

    Returns:
        List of tracked PIDs for the server
    """
    from .server_configuration import get_host_variants

    server_keys = get_host_variants(host, port)

    pids = []
    for key in server_keys:
        if key in servers:
            pids.extend(servers[key])

    return pids
