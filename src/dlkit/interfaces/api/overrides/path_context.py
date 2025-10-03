"""Path override context for API calls.

This module provides a way to pass API path overrides to the path resolution
system without polluting global environment variables or breaking the single
responsibility principle of DLKitEnvironment.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Generator

from dlkit.tools.config.environment import DLKitEnvironment


@dataclass
class PathOverrideContext:
    """Context for API path overrides.

    This allows API calls to temporarily override path resolution
    without affecting the global environment configuration.
    """

    root_dir: Path | None = None
    output_dir: Path | None = None
    data_dir: Path | None = None
    checkpoints_dir: Path | None = None


# Thread-local storage for path override context
_context_storage = threading.local()


def get_current_path_context() -> PathOverrideContext | None:
    """Get the current path override context for this thread.

    Returns:
        PathOverrideContext or None if no context is active
    """
    return getattr(_context_storage, "path_context", None)


def set_path_context(context: PathOverrideContext | None) -> None:
    """Set the path override context for this thread.

    Args:
        context: Path override context or None to clear
    """
    _context_storage.path_context = context


@contextmanager
def path_override_context(overrides: dict[str, Any]) -> Generator[None]:
    """Context manager for applying API path overrides.

    Args:
        overrides: Dictionary of path overrides from API calls

    Yields:
        None
    """
    # Convert overrides to PathOverrideContext
    context = PathOverrideContext(
        root_dir=Path(overrides["root_dir"]) if overrides.get("root_dir") else None,
        output_dir=Path(overrides["output_dir"]) if overrides.get("output_dir") else None,
        data_dir=Path(overrides["data_dir"]) if overrides.get("data_dir") else None,
        checkpoints_dir=Path(overrides["checkpoints_dir"])
        if overrides.get("checkpoints_dir")
        else None,
    )

    # Store the previous context
    previous_context = get_current_path_context()

    try:
        # Set the new context
        set_path_context(context)
        yield
    finally:
        # Restore the previous context
        set_path_context(previous_context)


def resolve_with_context(component_path: str, env: DLKitEnvironment | None = None) -> Path:
    """Resolve a component path with current override context.

    Args:
        component_path: Path to resolve (e.g., "output/mlruns")
        env: Environment instance (uses global if None)

    Returns:
        Resolved path respecting current override context
    """
    # Get current override context
    context = get_current_path_context()

    # Determine effective root directory
    if context and context.root_dir:
        effective_root = context.root_dir
    elif env is not None:
        effective_root = env.get_root_path()
    else:
        # Fallback to current working directory when no override is provided
        effective_root = Path.cwd()

    # Handle specific component path overrides
    if context:
        # Check for direct component overrides
        if component_path == "output" and context.output_dir:
            return context.output_dir.resolve()
        elif component_path.startswith("output/") and context.output_dir:
            # For paths like "output/mlruns", replace "output" with the override
            relative_path = component_path[7:]  # Remove "output/"
            return (context.output_dir / relative_path).resolve()
        elif component_path == "dataflow" and context.data_dir:
            return context.data_dir.resolve()
        elif component_path.startswith("dataflow/") and context.data_dir:
            relative_path = component_path[5:]  # Remove "dataflow/"
            return (context.data_dir / relative_path).resolve()
        elif "checkpoint" in component_path and context.checkpoints_dir:
            return context.checkpoints_dir.resolve()

    # Default resolution using environment root
    from dlkit.tools.io.resolution.factory import create_default_resolver_system

    registry, resolver_context = create_default_resolver_system(effective_root)
    resolved = registry.resolve(component_path, resolver_context)

    return Path(resolved) if resolved is not None else effective_root / component_path
