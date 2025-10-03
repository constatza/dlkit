"""Storage setup adapter implementing single responsibility principle."""

from __future__ import annotations

from typing import Any

from .domain_protocols import StorageSetup, UserInteraction, FileSystemOperations
from .domain_functions import should_use_default_storage, get_default_mlruns_path


class MLflowStorageSetup(StorageSetup):
    """MLflow storage setup implementation (SRP: Only handles storage configuration)."""

    def __init__(
        self, user_interaction: UserInteraction, file_system: FileSystemOperations
    ) -> None:
        """Initialize with dependencies.

        Args:
            user_interaction: User interaction handler
            file_system: File system operations handler
        """
        self._user_interaction = user_interaction
        self._file_system = file_system

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
        mlruns_path = get_default_mlruns_path()

        if not self._file_system.directory_exists(mlruns_path):
            self._file_system.create_directory(mlruns_path)

        return server_config
