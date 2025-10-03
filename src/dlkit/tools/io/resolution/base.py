"""Abstract base classes for URL and path resolution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from pathlib import Path

if TYPE_CHECKING:
    from .context import ResolverContext


class URLResolver(ABC):
    """Abstract base class for URL resolution strategies.

    Handles URLs with schemes (file://, sqlite://, http://, etc.) using pydantic
    URL types for proper parsing, validation, and manipulation.
    """

    @abstractmethod
    def resolve(self, url_string: str, context: ResolverContext) -> str:
        """Resolve URL with proper handling of paths and schemes.

        Args:
            url_string: URL string to resolve
            context: Resolution context with root paths and configuration

        Returns:
            str: Resolved URL string with proper path expansion and validation
        """
        pass


class PathResolver(ABC):
    """Abstract base class for path resolution strategies.

    Handles non-URL path values using pathlib.Path for proper file system
    operations and path normalization.
    """

    @abstractmethod
    def resolve(self, value: Any, context: ResolverContext) -> Path:
        """Resolve path value using pathlib operations.

        Args:
            value: Path value to resolve (str or Path object)
            context: Resolution context with root paths and configuration

        Returns:
            Path: Resolved absolute path with proper expansion
        """
        pass
