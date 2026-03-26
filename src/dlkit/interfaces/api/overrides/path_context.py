"""Path override context for API calls.

Re-exports thread-local context primitives from tools.io.path_context.
Keeps resolve_with_context locally as it needs DLKitEnvironment.
"""

from __future__ import annotations

from pathlib import Path

from dlkit.tools.io.path_context import (
    PathOverrideContext,
    get_current_path_context,
    path_override_context,
    set_path_context,
)

__all__ = [
    "PathOverrideContext",
    "get_current_path_context",
    "path_override_context",
    "resolve_with_context",
    "set_path_context",
]


def resolve_with_context(component_path: str, env=None) -> Path:
    """Resolve a component path with current override context.

    Args:
        component_path: Path to resolve (e.g., "output/mlruns")
        env: Environment instance (uses global if None)

    Returns:
        Resolved path respecting current override context
    """
    from loguru import logger

    from dlkit.tools.config.environment import env as global_env
    from dlkit.tools.io.resolution.factory import create_default_resolver_system

    effective_env = env if env is not None else global_env

    context = get_current_path_context()

    if context and context.root_dir:
        effective_root = context.root_dir
        logger.debug(f"resolve_with_context: Using PathOverrideContext root: {effective_root}")
    else:
        effective_root = effective_env.get_root_path()
        logger.debug(f"resolve_with_context: Using DLKitEnvironment root: {effective_root}")

    if context:
        if component_path == "output" and context.output_dir:
            return context.output_dir.resolve()
        if component_path.startswith("output/") and context.output_dir:
            relative_path = component_path[7:]
            return (context.output_dir / relative_path).resolve()
        if component_path == "dataflow" and context.data_dir:
            return context.data_dir.resolve()
        if component_path.startswith("dataflow/") and context.data_dir:
            relative_path = component_path[5:]
            return (context.data_dir / relative_path).resolve()
        if "checkpoint" in component_path and context.checkpoints_dir:
            return context.checkpoints_dir.resolve()

    registry, resolver_context = create_default_resolver_system(effective_root)
    resolved = registry.resolve(component_path, resolver_context)
    return Path(resolved) if resolved is not None else (effective_root / component_path).resolve()
