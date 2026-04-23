"""Thread-local path override context for DLKit path resolution.

Provides lightweight thread-safe context storage for path overrides.
No dependency on tools/config or interfaces layers.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dlkit.infrastructure.io.explicit_path_context import (
    PathContext,
    resolve_component_path,
    resolve_root_dir,
)
from dlkit.infrastructure.io.path_context_state import (
    PathOverrideContext,
    get_current_path_context,
    set_path_context,
)
from dlkit.infrastructure.io.paths import normalize_user_path


@contextmanager
def path_override_context(overrides: dict[str, Any]) -> Generator[None]:
    """Context manager for applying API path overrides.

    Args:
        overrides: Dictionary of path overrides from API calls

    Yields:
        None
    """
    root_dir = normalize_user_path(overrides.get("root_dir"), require_absolute=True)
    output_dir = normalize_user_path(overrides.get("output_dir"))
    data_dir = normalize_user_path(overrides.get("data_dir"))
    checkpoints_dir = normalize_user_path(overrides.get("checkpoints_dir"))

    context = PathOverrideContext(
        root_dir=root_dir,
        output_dir=output_dir,
        data_dir=data_dir,
        checkpoints_dir=checkpoints_dir,
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


def resolve_with_context(component_path: str, env=None) -> Path:
    """Resolve a component path with current override context.

    This function uses the unified PathResolver which consolidates path
    resolution logic and provides consistent precedence handling.

    Args:
        component_path: Path to resolve (e.g., "output/mlruns")
        env: Environment instance (uses global if None, currently unused
            as PathResolver reads from global defaults)

    Returns:
        Resolved path respecting current override context
    """
    from dlkit.infrastructure.io.path_resolver import PathResolver

    resolver = PathResolver.from_defaults()
    return resolver.resolve_component_path(component_path)


__all__ = [
    "PathContext",
    "PathOverrideContext",
    "get_current_path_context",
    "path_override_context",
    "resolve_component_path",
    "resolve_root_dir",
    "resolve_with_context",
    "set_path_context",
]
