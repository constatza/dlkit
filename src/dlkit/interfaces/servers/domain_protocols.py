"""Domain protocols for server management following SOLID principles."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Protocol


class ServerTracker(Protocol):
    """Protocol for tracking server instances (SRP: Single responsibility for tracking)."""

    @abstractmethod
    def track_server(self, host: str, port: int, pid: int) -> None:
        """Track a server instance.

        Args:
            host: Server hostname
            port: Server port
            pid: Process ID
        """

    @abstractmethod
    def untrack_server(self, host: str, port: int) -> None:
        """Remove server from tracking.

        Args:
            host: Server hostname
            port: Server port
        """

    @abstractmethod
    def get_tracked_pids(self, host: str, port: int) -> list[int]:
        """Get tracked PIDs for a server.

        Args:
            host: Server hostname
            port: Server port

        Returns:
            List of tracked process IDs
        """


class ProcessKiller(Protocol):
    """Protocol for terminating server processes (SRP: Single responsibility for process termination)."""

    @abstractmethod
    def stop_server_processes(self, host: str, port: int, force: bool = False) -> bool:
        """Stop server processes for given host:port.

        Args:
            host: Server hostname
            port: Server port
            force: Whether to force kill processes

        Returns:
            True if stopped successfully or no processes found
            False if operational failure

        Raises:
            RuntimeError: If exceptional failure

        Original Returns:
            Tuple of (success, status_messages)
        """


class StorageSetup(Protocol):
    """Protocol for storage configuration (SRP: Single responsibility for storage setup)."""

    @abstractmethod
    def ensure_storage_setup(self, server_config: Any, overrides: dict[str, Any]) -> Any:
        """Ensure storage locations are properly configured.

        This is pure business logic that creates required storage.
        User interaction should be handled at the CLI layer.

        Args:
            server_config: Server configuration object
            overrides: CLI parameter overrides

        Returns:
            Updated server configuration
        """


class UserInteraction(Protocol):
    """Protocol for user interactions (SRP: Single responsibility for UI)."""

    @abstractmethod
    def confirm_action(
        self, message: str, default: bool = True, auto_confirm: bool = False
    ) -> bool:
        """Ask user for confirmation.

        Args:
            message: Confirmation prompt
            default: Default choice
            auto_confirm: If True, return default without prompting

        Returns:
            User's choice
        """

    @abstractmethod
    def show_message(self, message: str) -> None:
        """Display message to user.

        Args:
            message: Message to display
        """


class FileSystemOperations(Protocol):
    """Protocol for file system operations (SRP: Single responsibility for file I/O)."""

    @abstractmethod
    def create_directory(self, path: Path) -> None:
        """Create directory if it doesn't exist.

        Args:
            path: Directory path to create
        """

    @abstractmethod
    def directory_exists(self, path: Path) -> bool:
        """Check if directory exists.

        Args:
            path: Directory path to check

        Returns:
            True if directory exists
        """


class ServerContextFactory(Protocol):
    """Protocol for creating server contexts (SRP: Factory responsibility)."""

    @abstractmethod
    def create_server_context(self, mlflow_settings: Any, **overrides: Any) -> Any:
        """Create MLflow server context from settings.

        Args:
            mlflow_settings: MLflow configuration settings
            **overrides: Configuration overrides

        Returns:
            Server context ready for use
        """
