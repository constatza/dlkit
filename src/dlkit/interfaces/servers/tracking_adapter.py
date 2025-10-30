"""Server tracking adapter implementing single responsibility principle."""

from __future__ import annotations

from dlkit.tools.utils.logging_config import get_logger
from dlkit.tools.config.environment import DLKitEnvironment

from .server_tracking_operations import (
    load_tracking_data,
    save_tracking_data,
    add_server_to_tracking,
    remove_server_from_tracking,
    get_pids_for_server,
)
from .path_resolution import ServerPathResolver
from .domain_protocols import ServerTracker

logger = get_logger(__name__)


class FileBasedServerTracker(ServerTracker):
    """File-based implementation of server tracking (SRP: Only handles tracking)."""

    def __init__(self, path_resolver: ServerPathResolver | None = None) -> None:
        """Initialize with optional path resolver dependency.

        Args:
            path_resolver: Path resolver for tracking file location (creates default if None)
        """
        self._path_resolver = path_resolver or ServerPathResolver(DLKitEnvironment())

    def track_server(self, host: str, port: int, pid: int) -> None:
        """Track a server instance by persisting to file.

        Args:
            host: Server hostname
            port: Server port
            pid: Process ID
        """
        try:
            tracking_file = self._path_resolver.get_tracking_file_path()
            servers = load_tracking_data(tracking_file)
            updated_servers = add_server_to_tracking(servers, host, port, pid)
            save_tracking_data(tracking_file, updated_servers)
        except Exception as e:
            # Non-critical: tracking failure doesn't break server startup
            logger.debug(
                f"Non-critical: Failed to track server {host}:{port} (PID {pid}) - {e}",
                exc_info=True
            )

    def untrack_server(self, host: str, port: int) -> None:
        """Remove server from tracking file.

        Args:
            host: Server hostname
            port: Server port
        """
        try:
            tracking_file = self._path_resolver.get_tracking_file_path()
            if not tracking_file.exists():
                return

            servers = load_tracking_data(tracking_file)
            updated_servers = remove_server_from_tracking(servers, host, port)
            save_tracking_data(tracking_file, updated_servers)
        except Exception as e:
            # Non-critical: tracking cleanup failure is not fatal
            logger.debug(
                f"Non-critical: Failed to untrack server {host}:{port} - {e}",
                exc_info=True
            )

    def get_tracked_pids(self, host: str, port: int) -> list[int]:
        """Get PIDs of tracked servers for host:port.

        Args:
            host: Server hostname
            port: Server port

        Returns:
            List of tracked process IDs
        """
        try:
            tracking_file = self._path_resolver.get_tracking_file_path()
            if not tracking_file.exists():
                return []

            servers = load_tracking_data(tracking_file)
            return get_pids_for_server(servers, host, port)
        except Exception as e:
            # Non-critical: return empty list if tracking data unavailable
            logger.debug(
                f"Non-critical: Failed to get tracked PIDs for {host}:{port} - {e}",
                exc_info=True
            )
            return []
