"""Factory functions for creating resolver instances."""

from __future__ import annotations

from pathlib import Path

from .registry import ResolverRegistry
from .context import ResolverContext


def create_resolver_registry() -> ResolverRegistry:
    """Factory for creating resolver registry with default configuration.

    Returns:
        ResolverRegistry: Registry with generic URL and path resolvers
    """
    return ResolverRegistry()


def create_resolver_context(
    root_path: Path, home_path: Path | None = None, additional_file_schemes: set[str] | None = None
) -> ResolverContext:
    """Factory for creating resolver context.

    Args:
        root_path: Base directory for resolving relative paths
        home_path: User home directory (defaults to Path.home())
        additional_file_schemes: Extra schemes to treat as file-path based

    Returns:
        ResolverContext: Context with specified configuration
    """
    # Start with default context
    context = ResolverContext(root_path=root_path, home_path=home_path or Path.home())

    # Add any additional file path schemes
    if additional_file_schemes:
        for scheme in additional_file_schemes:
            context = context.add_file_path_scheme(scheme)

    return context


def create_default_resolver_system(root_path: Path) -> tuple[ResolverRegistry, ResolverContext]:
    """Factory for creating complete resolver system.

    Convenience function that creates both registry and context with
    sensible defaults for most use cases.

    Args:
        root_path: Base directory for resolving relative paths

    Returns:
        tuple: (ResolverRegistry, ResolverContext) ready for use
    """
    registry = create_resolver_registry()
    context = create_resolver_context(root_path)
    return registry, context
