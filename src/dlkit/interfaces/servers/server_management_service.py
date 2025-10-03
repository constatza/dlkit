"""Unified server management service following dependency inversion principle."""

from __future__ import annotations

from typing import Any

from .domain_protocols import ServerTracker, ProcessKiller, StorageSetup, ServerContextFactory
from .tracking_adapter import FileBasedServerTracker
from .process_adapter import PsutilProcessKiller
from .storage_adapter import MLflowStorageSetup
from .infrastructure_adapters import (
    TyperUserInteraction,
    StandardFileSystemOperations,
    MLflowContextFactory,
)


class ServerManagementService:
    """Unified server management service (DIP: Depends on abstractions, not implementations)."""

    def __init__(
        self,
        server_tracker: ServerTracker | None = None,
        process_killer: ProcessKiller | None = None,
        storage_setup: StorageSetup | None = None,
        context_factory: ServerContextFactory | None = None,
    ) -> None:
        """Initialize with dependency injection.

        Args:
            server_tracker: Server tracking implementation
            process_killer: Process killing implementation
            storage_setup: Storage setup implementation
            context_factory: Server context factory implementation
        """
        # Default implementations if not provided (OCP: Open for extension)
        self._server_tracker = server_tracker or FileBasedServerTracker()
        self._process_killer = process_killer or PsutilProcessKiller(self._server_tracker)

        # Infrastructure dependencies
        user_interaction = TyperUserInteraction()
        file_system = StandardFileSystemOperations()

        self._storage_setup = storage_setup or MLflowStorageSetup(user_interaction, file_system)
        self._context_factory = context_factory or MLflowContextFactory()

    def track_server(self, host: str, port: int, pid: int) -> None:
        """Track a server instance.

        Args:
            host: Server hostname
            port: Server port
            pid: Process ID
        """
        self._server_tracker.track_server(host, port, pid)

    def untrack_server(self, host: str, port: int) -> None:
        """Remove server from tracking.

        Args:
            host: Server hostname
            port: Server port
        """
        self._server_tracker.untrack_server(host, port)

    def get_tracked_pids(self, host: str, port: int) -> list[int]:
        """Get tracked PIDs for a server.

        Args:
            host: Server hostname
            port: Server port

        Returns:
            List of tracked process IDs
        """
        return self._server_tracker.get_tracked_pids(host, port)

    def stop_server_processes(
        self, host: str, port: int, force: bool = False
    ) -> tuple[bool, list[str]]:
        """Stop server processes for given host:port.

        Args:
            host: Server hostname
            port: Server port
            force: Whether to force kill processes

        Returns:
            Tuple of (success, status_messages)
        """
        return self._process_killer.stop_server_processes(host, port, force)

    def ensure_storage_setup(self, server_config: Any, overrides: dict[str, Any]) -> Any:
        """Ensure storage locations are properly configured.

        This is pure business logic - user interaction is handled at the CLI layer.

        Args:
            server_config: Server configuration object
            overrides: CLI parameter overrides

        Returns:
            Updated server configuration
        """
        return self._storage_setup.ensure_storage_setup(server_config, overrides)

    def create_server_context(self, mlflow_settings: Any, **overrides: Any) -> Any:
        """Create server context from settings.

        Args:
            mlflow_settings: MLflow configuration settings
            **overrides: Configuration overrides

        Returns:
            Server context ready for use
        """
        return self._context_factory.create_server_context(mlflow_settings, **overrides)
