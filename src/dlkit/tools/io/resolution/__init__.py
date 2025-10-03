"""Unified URL and path resolution system following SOLID principles."""

from .base import URLResolver, PathResolver
from .context import ResolverContext
from .registry import ResolverRegistry
from .path_resolver import GenericPathResolver
from .factory import (
    create_resolver_registry,
    create_resolver_context,
    create_default_resolver_system,
)

__all__ = [
    "URLResolver",
    "PathResolver",
    "ResolverContext",
    "ResolverRegistry",
    "GenericPathResolver",
    "create_resolver_registry",
    "create_resolver_context",
    "create_default_resolver_system",
]
