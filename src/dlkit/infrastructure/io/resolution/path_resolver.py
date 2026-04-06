"""Generic path resolver using pathlib.Path for all path operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import PathResolver
from .context import ResolverContext


class GenericPathResolver(PathResolver):
    """Generic path resolver handling all path types using pathlib.Path.

    Handles tilde expansion, relative path resolution, and path normalization
    using pathlib.Path for proper cross-platform file system operations.

    Specifically designed for non-URL path values (strings and Path objects
    without schemes).
    """

    def resolve(self, value: Any, context: ResolverContext) -> Path:
        """Generic path resolution using pathlib operations.

        Args:
            value: Path value to resolve (str or Path object)
            context: Resolution context with root path information

        Returns:
            Path: Resolved absolute path with proper expansion

        Raises:
            ValueError: If value is not a valid path type or contains URL scheme
        """
        if not isinstance(value, (str, Path)):
            raise ValueError(f"Cannot resolve non-path value: {type(value).__name__} = {value}")

        path_str = str(value)

        # URLs should be handled by URL resolver, not path resolver
        if "://" in path_str:
            raise ValueError(f"URLs should use URL resolver: {path_str}")

        path = Path(path_str)

        # Handle tilde expansion using context home path
        if "~" in path_str:
            if path_str.startswith("~/"):
                # Replace ~/... with context home path
                path = context.home_path / path_str[2:]
            elif path_str == "~":
                # Just ~ becomes home path
                path = context.home_path
            else:
                # For other tilde cases, use expanduser as fallback
                path = path.expanduser()

        # Handle relative path resolution
        if not path.is_absolute():
            path = context.root_path / path

        # Resolve to normalize the path (remove .., ., symlinks, etc.)
        return path.resolve()

    def can_resolve(self, value: Any) -> bool:
        """Check if this resolver can handle the given value.

        Args:
            value: Value to check

        Returns:
            bool: True if value is a resolvable path
        """
        if not isinstance(value, (str, Path)):
            return False

        # Skip URLs
        if "://" in str(value):
            return False

        # Try to create a Path object to validate
        try:
            Path(str(value))
            return True
        except Exception:
            return False
