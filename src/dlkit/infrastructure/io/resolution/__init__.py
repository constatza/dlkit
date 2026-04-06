"""Unified URL and path resolution system following SOLID principles."""

from .base import PathResolver, URLResolver
from .context import ResolverContext
from .factory import (
    create_default_resolver_system,
    create_resolver_context,
    create_resolver_registry,
)
from .path_resolver import GenericPathResolver
from .registry import ResolverRegistry

__all__ = [
    "GenericPathResolver",
    "PathResolver",
    "ResolverContext",
    "ResolverRegistry",
    "URLResolver",
    "create_default_resolver_system",
    "create_resolver_context",
    "create_resolver_registry",
]
