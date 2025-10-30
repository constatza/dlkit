"""Path resolution for server components with dependency injection.

This module provides centralized path resolution for server-related
components (tracking files, storage directories, etc.) with proper
dependency injection of DLKitEnvironment.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dlkit.tools.io import locations

if TYPE_CHECKING:  # pragma: no cover
    from dlkit.tools.config.environment import DLKitEnvironment


class ServerPathResolver:
    """Resolve server-related paths with injected environment dependency.

    This class follows the Dependency Inversion Principle by injecting
    DLKitEnvironment rather than using a global instance. All methods
    are stateless and pure (given the same environment, they return the
    same paths).

    Args:
        env: DLKitEnvironment instance for root path resolution
    """

    def __init__(self, env: DLKitEnvironment) -> None:
        """Initialize path resolver with environment dependency.

        Args:
            env: DLKitEnvironment instance for path resolution
        """
        self._env = env

    def get_tracking_file_path(self) -> Path:
        """Get path to server tracking file under the user's home directory.

        Returns:
            Path to ~/.dlkit/servers.json
        """
        return self._env.get_server_tracking_path()

    def resolve_component_path(self, path_value: str | Path | None) -> Path | None:
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

            return resolve_with_context(str(path_value), self._env)
        except ImportError:
            # Fallback to direct resolution if path_context is not available
            from dlkit.tools.io.resolution.factory import create_default_resolver_system

            registry, context = create_default_resolver_system(self._env.get_root_path())
            resolved = registry.resolve(path_value, context)

            # Ensure we return a Path object
            return Path(resolved) if resolved is not None else None

    def get_default_output_dir(self) -> Path:
        """Get default output directory with environment-aware root resolution.

        Returns:
            Path to default output directory under environment root
        """
        # Use centralized locations policy
        return locations.output()

    def get_default_mlruns_path(self) -> Path:
        """Get default MLruns path with environment-aware root resolution.

        Returns:
            Path to default MLruns directory under output/
        """
        # Backward-compatible default under output/
        return locations.output("mlruns")

    def get_default_optuna_storage_url(self) -> str:
        """Get default Optuna storage URL with environment-aware root resolution.

        Returns:
            Default Optuna storage URL under output/
        """
        # Centralized default storage URI
        return locations.optuna_storage_uri()
