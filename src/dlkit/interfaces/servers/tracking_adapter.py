"""Server tracking adapter implementing single responsibility principle."""

from __future__ import annotations

from .domain_protocols import ServerTracker
from .domain_functions import (
    get_tracking_file_path,
    load_tracking_data,
    save_tracking_data,
    add_server_to_tracking,
    remove_server_from_tracking,
    get_pids_for_server,
)


class FileBasedServerTracker(ServerTracker):
    """File-based implementation of server tracking (SRP: Only handles tracking)."""

    def track_server(self, host: str, port: int, pid: int) -> None:
        """Track a server instance by persisting to file.

        Args:
            host: Server hostname
            port: Server port
            pid: Process ID
        """
        try:
            tracking_file = get_tracking_file_path()
            servers = load_tracking_data(tracking_file)
            updated_servers = add_server_to_tracking(servers, host, port, pid)
            save_tracking_data(tracking_file, updated_servers)
        except Exception:
            # Silent failure for tracking - don't break server startup
            pass

    def untrack_server(self, host: str, port: int) -> None:
        """Remove server from tracking file.

        Args:
            host: Server hostname
            port: Server port
        """
        try:
            tracking_file = get_tracking_file_path()
            if not tracking_file.exists():
                return

            servers = load_tracking_data(tracking_file)
            updated_servers = remove_server_from_tracking(servers, host, port)
            save_tracking_data(tracking_file, updated_servers)
        except Exception:
            # Silent failure for tracking
            pass

    def get_tracked_pids(self, host: str, port: int) -> list[int]:
        """Get PIDs of tracked servers for host:port.

        Args:
            host: Server hostname
            port: Server port

        Returns:
            List of tracked process IDs
        """
        try:
            tracking_file = get_tracking_file_path()
            if not tracking_file.exists():
                return []

            servers = load_tracking_data(tracking_file)
            return get_pids_for_server(servers, host, port)
        except Exception:
            return []
