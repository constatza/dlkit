"""Server management protocols and domain models."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol


class ServerInfo:
    """Information about a running server instance."""

    def __init__(
        self,
        process: Any,
        url: str,
        host: str,
        port: int,
        pid: int | None = None,
    ) -> None:
        """Initialize server info.

        Args:
            process: Server process handle
            url: Full server URL
            host: Server hostname
            port: Server port
            pid: Process ID if available
        """
        self.process = process
        self.url = url
        self.host = host
        self.port = port
        self.pid = pid or (process.pid if hasattr(process, "pid") else None)

    def __repr__(self) -> str:
        """String representation."""
        return f"ServerInfo(url='{self.url}', pid={self.pid})"


class ServerStatus:
    """Status information about a server."""

    def __init__(
        self,
        is_running: bool,
        url: str,
        response_time: float | None = None,
        error_message: str | None = None,
    ) -> None:
        """Initialize server status.

        Args:
            is_running: Whether server is responding
            url: Server URL that was checked
            response_time: Response time in seconds if available
            error_message: Error message if server is not responding
        """
        self.is_running = is_running
        self.url = url
        self.response_time = response_time
        self.error_message = error_message

    def __repr__(self) -> str:
        """String representation."""
        status = "running" if self.is_running else "not running"
        return f"ServerStatus({status} at {self.url})"


class ProcessManager(Protocol):
    """Protocol for managing server processes."""

    @abstractmethod
    def start_process(self, config: Any) -> Any:
        """Start a server process.

        Args:
            config: Server configuration

        Returns:
            Process handle
        """

    @abstractmethod
    def stop_process(self, process: Any) -> bool:
        """Stop a server process.

        Args:
            process: Process handle to stop

        Returns:
            True if stopped successfully
        """

    @abstractmethod
    def is_process_running(self, process: Any) -> bool:
        """Check if a process is still running.

        Args:
            process: Process handle to check

        Returns:
            True if process is running
        """


class HealthChecker(Protocol):
    """Protocol for checking server health."""

    @abstractmethod
    def check_health(self, url: str, timeout: float = 5.0) -> ServerStatus:
        """Check server health at given URL.

        Args:
            url: Server URL to check
            timeout: Request timeout in seconds

        Returns:
            ServerStatus with health information
        """

    @abstractmethod
    def wait_for_health(self, url: str, timeout: float = 10.0, poll_interval: float = 0.5) -> bool:
        """Wait for server to become healthy.

        Args:
            url: Server URL to check
            timeout: Maximum time to wait in seconds
            poll_interval: Time between checks in seconds

        Returns:
            True if server became healthy within timeout
        """


class ServerAdapter(Protocol):
    """Protocol for server management with proper separation of concerns."""

    @abstractmethod
    def start_server(
        self,
        server_config: Any,
        **overrides: Any,
    ) -> ServerInfo:
        """Start a server with optional configuration overrides.

        Args:
            server_config: Server-specific configuration
            **overrides: Server configuration overrides

        Returns:
            ServerInfo containing server details
        """

    @abstractmethod
    def stop_server(self, server_info: ServerInfo) -> bool:
        """Stop a running server.

        Args:
            server_info: Information about the server to stop

        Returns:
            True if stopped successfully
        """

    @abstractmethod
    def check_server(
        self,
        host: str = "localhost",
        port: int = 5000,
    ) -> ServerStatus:
        """Check if a server is running and responding.

        Args:
            host: Server hostname to check
            port: Server port to check

        Returns:
            ServerStatus for the given host/port
        """

    @abstractmethod
    def get_server_url(self, host: str, port: int) -> str:
        """Get server URL from host and port.

        Args:
            host: Server hostname
            port: Server port

        Returns:
            Complete server URL
        """


class ContextualServerAdapter(ServerAdapter, Protocol):
    """Server adapter that can be used as a context manager."""

    def __enter__(self) -> ServerInfo:
        """Enter context and return server info."""
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and clean up server."""
        ...
