"""Storage setup adapter implementing single responsibility principle."""

from __future__ import annotations

from typing import Any

from dlkit.tools.config.environment import DLKitEnvironment

from .domain_protocols import StorageSetup, UserInteraction, FileSystemOperations
from .server_configuration import should_use_default_storage
from .path_resolution import ServerPathResolver


class MLflowStorageSetup(StorageSetup):
    """MLflow storage setup implementation (SRP: Only handles storage configuration)."""

    def __init__(
        self,
        user_interaction: UserInteraction,
        file_system: FileSystemOperations,
        path_resolver: ServerPathResolver | None = None,
    ) -> None:
        """Initialize with dependencies.

        Args:
            user_interaction: User interaction handler
            file_system: File system operations handler
            path_resolver: Path resolver for default locations (creates default if None)
        """
        self._user_interaction = user_interaction
        self._file_system = file_system
        self._path_resolver = path_resolver or ServerPathResolver(DLKitEnvironment())

    def ensure_storage_setup(self, server_config: Any, overrides: dict[str, Any]) -> Any:
        """Ensure MLflow storage locations are properly configured.

        This is pure business logic - it creates required storage without user interaction.
        The CLI layer should handle user decisions before calling this method.

        Args:
            server_config: Server configuration object
            overrides: CLI parameter overrides

        Returns:
            Updated server configuration
        """
        if not should_use_default_storage(server_config, overrides):
            return server_config

        # Check if default mlruns directory exists and create it if needed
        mlruns_path = self._path_resolver.get_default_mlruns_path()

        if not self._file_system.directory_exists(mlruns_path):
            self._file_system.create_directory(mlruns_path)

        return server_config
