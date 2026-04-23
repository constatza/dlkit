"""Unified path resolver service.

This module provides a single, centralized PathResolver service that handles
all path resolution in DLKit. It consolidates the thread-local PathOverrideContext
and global EnvironmentSettings into a single decision point with clear precedence.

Design principles:
- Single source of truth for path resolution logic
- Clear, explicit precedence: context > env_var > config > cwd
- Dependency injection (no direct reads from thread-local or global state)
- Pure function semantics given inputs
- Type-safe with comprehensive hints
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dlkit.infrastructure.config.environment import EnvironmentSettings
    from dlkit.infrastructure.io.path_context_state import PathOverrideContext


class PathResolver:
    """Unified path resolver with explicit precedence.

    This class centralizes all path resolution logic and implements a single,
    clear precedence hierarchy:

    Priority (root_dir resolution):
        1. thread_local_context.root_dir (API/CLI override)
        2. DLKIT_ROOT_DIR env var (via environment settings)
        3. config_root_dir from SESSION.root_dir
        4. Path.cwd() (current working directory fallback)

    Attributes:
        _thread_local_context: Thread-local path override context
            (may be None when not set)
        _environment: Global environment settings
        _config_root: Optional root_dir from loaded config (SESSION.root_dir)
    """

    __slots__ = ("_thread_local_context", "_environment", "_config_root")

    def __init__(
        self,
        thread_local_context: PathOverrideContext | None = None,
        environment: EnvironmentSettings | None = None,
        config_root: Path | str | None = None,
    ) -> None:
        """Initialize the path resolver with explicit inputs.

        Args:
            thread_local_context: Thread-local path override context.
                Typically from get_current_path_context(). May be None.
            environment: Global environment settings. Typically the global
                EnvironmentSettings instance. May be None.
            config_root: Optional root_dir from loaded config (SESSION.root_dir).
                May be None or a str/Path.
        """
        self._thread_local_context = thread_local_context
        self._environment = environment
        self._config_root = Path(config_root) if config_root else None

    @classmethod
    def from_defaults(cls) -> PathResolver:
        """Create a PathResolver using global defaults.

        This factory method reads from the current thread-local context and
        global EnvironmentSettings. Useful when you don't have explicit
        context/environment references.

        Returns:
            PathResolver instance configured with current defaults.
        """
        from dlkit.infrastructure.config.environment import env as global_env
        from dlkit.infrastructure.io.path_context_state import get_current_path_context

        return cls(
            thread_local_context=get_current_path_context(),
            environment=global_env,
        )

    def resolve(self, path: Path | str | None = None) -> Path:
        """Resolve a path with the unified precedence hierarchy.

        This is the core resolution function. It resolves either a specific
        path or the root directory itself, applying precedence rules.

        Args:
            path: Path to resolve. If None, resolves to root directory.
                Relative paths are resolved relative to root.
                Absolute paths are returned as-is.

        Returns:
            Resolved absolute Path.

        Example:
            >>> resolver = PathResolver.from_defaults()
            >>> root = resolver.resolve()  # Get root directory
            >>> data_dir = resolver.resolve("data")  # Get data subdirectory
        """
        # Resolve root directory first
        root_dir = self._resolve_root()

        # If no path specified, return root
        if path is None:
            return root_dir.resolve()

        path_obj = Path(path)

        # Absolute paths bypass root resolution
        if path_obj.is_absolute():
            return path_obj.resolve()

        # Relative paths are resolved relative to root
        return (root_dir / path_obj).resolve()

    def resolve_component_path(self, component_path: str) -> Path:
        """Resolve a component path (e.g., 'output/mlruns').

        This method handles special DLKit component paths, checking for
        specific overrides in the thread-local context before falling back
        to standard root-relative resolution.

        Args:
            component_path: Component path. Examples:
                - "output": Output directory
                - "output/mlruns": MLflow tracking directory
                - "output/predictions": Predictions directory
                - "data": Data directory
                - "checkpoints": Checkpoints directory

        Returns:
            Resolved absolute path for the component.

        Example:
            >>> resolver = PathResolver.from_defaults()
            >>> mlruns_path = resolver.resolve_component_path("output/mlruns")
        """
        # Check for direct component overrides in thread-local context
        if self._thread_local_context:
            # Direct output directory override
            if component_path == "output" and self._thread_local_context.output_dir:
                return self._thread_local_context.output_dir.resolve()

            # Prefixed output paths (e.g., "output/mlruns")
            if component_path.startswith("output/") and self._thread_local_context.output_dir:
                relative_path = component_path[7:]  # Remove "output/" prefix
                return (self._thread_local_context.output_dir / relative_path).resolve()

            # Direct data directory override
            if component_path == "data" and self._thread_local_context.data_dir:
                return self._thread_local_context.data_dir.resolve()

            # Prefixed data paths
            if component_path.startswith("data/") and self._thread_local_context.data_dir:
                relative_path = component_path[5:]  # Remove "data/" prefix
                return (self._thread_local_context.data_dir / relative_path).resolve()

            # Checkpoints override (matches "checkpoints" or contains "checkpoint")
            if "checkpoint" in component_path and self._thread_local_context.checkpoints_dir:
                return self._thread_local_context.checkpoints_dir.resolve()

        # Fallback to standard resolution: root / component
        root_dir = self._resolve_root()
        return (root_dir / component_path).resolve()

    def _resolve_root(self) -> Path:
        """Resolve root directory with the unified precedence hierarchy.

        Priority:
            1. thread_local_context.root_dir
            2. environment.root_dir (from DLKIT_ROOT_DIR env var)
            3. config_root (from SESSION.root_dir)
            4. Path.cwd()

        Returns:
            Resolved root directory (always absolute).
        """
        # Priority 1: Thread-local context (explicit override)
        if self._thread_local_context and self._thread_local_context.root_dir:
            return self._thread_local_context.root_dir.resolve()

        # Priority 2: Environment (from DLKIT_ROOT_DIR env var or loaded SESSION.root_dir)
        if self._environment:
            env_root = self._environment.get_root_path()
            # Only use environment if it's not just the CWD (indicates it was explicitly set)
            if env_root != Path.cwd():
                return env_root.resolve()

        # Priority 3: Config-based root (from SESSION.root_dir)
        if self._config_root:
            if self._config_root.is_absolute():
                return self._config_root.resolve()
            else:
                return (Path.cwd() / self._config_root).resolve()

        # Priority 4: Fallback to CWD
        return Path.cwd().resolve()

    def get_root(self) -> Path:
        """Get the root directory (alias for resolve()).

        Returns:
            Resolved root directory.
        """
        return self._resolve_root().resolve()

    def has_context_override(self) -> bool:
        """Check if thread-local context has a root override.

        Returns:
            True if context.root_dir is explicitly set.
        """
        return (
            self._thread_local_context is not None
            and self._thread_local_context.root_dir is not None
        )

    def has_env_override(self) -> bool:
        """Check if environment has an explicit override.

        Returns:
            True if DLKIT_ROOT_DIR env var is set.
        """
        return os.environ.get("DLKIT_ROOT_DIR") is not None
