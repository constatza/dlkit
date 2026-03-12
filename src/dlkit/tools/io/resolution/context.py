"""Resolution context for URL and path operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolverContext:
    """Immutable context for resolution operations.

    Contains path information and scheme categorization for proper URL/path
    resolution across different protocols and file systems.

    Args:
        root_path: Base directory for resolving relative paths
        home_path: User home directory for tilde expansion
        file_path_schemes: URL schemes that contain file paths needing expansion
    """

    root_path: Path
    home_path: Path = field(default_factory=Path.home)

    # Simple scheme categorization based on path handling needs
    # These schemes contain file paths that need tilde/relative expansion
    file_path_schemes: set[str] = field(
        default_factory=lambda: {
            "file",  # file:///path/to/file
            "sqlite",  # sqlite:///path/to/database.db
        }
    )

    def is_file_path_scheme(self, scheme: str) -> bool:
        """Check if a URL scheme contains file paths needing expansion.

        Args:
            scheme: URL scheme (e.g., 'file', 'http', 'sqlite')

        Returns:
            bool: True if scheme contains file paths, False for network endpoints
        """
        return scheme.lower() in self.file_path_schemes

    def add_file_path_scheme(self, scheme: str) -> ResolverContext:
        """Create new context with additional file path scheme.

        Since context is immutable, this returns a new instance.

        Args:
            scheme: URL scheme to treat as file-path based

        Returns:
            ResolverContext: New context with added scheme
        """
        new_schemes = self.file_path_schemes | {scheme.lower()}
        return ResolverContext(
            root_path=self.root_path, home_path=self.home_path, file_path_schemes=new_schemes
        )
