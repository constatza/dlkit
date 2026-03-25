"""Simple registry with two generic resolvers."""

from __future__ import annotations

from typing import Any

from .context import ResolverContext
from .path_resolver import GenericPathResolver


class ResolverRegistry:
    """Simple registry with two generic resolvers.

    Provides a unified interface for resolving both URLs and paths through
    a single entry point. Automatically delegates to the appropriate resolver
    based on the input value type.
    """

    def __init__(self):
        """Initialize registry with generic resolvers."""
        self._url_resolver = None
        self._path_resolver = GenericPathResolver()

    def resolve(self, value: Any, context: ResolverContext) -> Any:
        """Single entry point - delegates based on value type.

        Args:
            value: Value to resolve (URL string, path string, or Path object)
            context: Resolution context with root paths and configuration

        Returns:
            Any: Resolved value maintaining input type (str -> str, Path -> Path)

        Raises:
            ValueError: If value cannot be resolved by either resolver
        """
        if value is None:
            return value

        # URL validation/resolution is handled by Pydantic types elsewhere.
        if isinstance(value, str) and ("://" in value or self._is_mlflow_edge_case(value)):
            return value

        # Path resolution for non-URL values
        try:
            resolved_path = self._path_resolver.resolve(value, context)
            # Return same type as input (str -> str, Path -> Path)
            return str(resolved_path) if isinstance(value, str) else resolved_path
        except ValueError:
            # If path resolution fails, return original value
            return value

    def resolve_url(self, url_string: str, context: ResolverContext) -> str:
        """Explicit URL resolution method.

        Args:
            url_string: URL string to resolve
            context: Resolution context

        Returns:
            str: Resolved URL string
        """
        # URL resolution removed from registry; return input for caller-side validation
        return url_string

    def resolve_path(self, path_value: Any, context: ResolverContext) -> Any:
        """Explicit path resolution method.

        Args:
            path_value: Path value to resolve
            context: Resolution context

        Returns:
            Any: Resolved path maintaining input type
        """
        try:
            resolved_path = self._path_resolver.resolve(path_value, context)
            return str(resolved_path) if isinstance(path_value, str) else resolved_path
        except ValueError:
            return path_value

    def _is_mlflow_edge_case(self, value: str) -> bool:
        """Check if value is an MLflow edge case (non-standard URI format)."""
        return (
            value == "databricks"
            or (value.startswith("databricks:") and "://" not in value)
            or (value.startswith("mlflow-artifacts:") and "://" not in value)
        )
