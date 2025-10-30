"""Infrastructure adapters for external dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .domain_protocols import UserInteraction, FileSystemOperations, ServerContextFactory


class TyperUserInteraction(UserInteraction):
    """Typer/Rich-based user interaction (SRP: Only handles user interface)."""

    def confirm_action(
        self, message: str, default: bool = True, auto_confirm: bool = False
    ) -> bool:
        """Ask user for confirmation using Typer.

        Args:
            message: Confirmation prompt
            default: Default choice
            auto_confirm: If True, return default without prompting

        Returns:
            User's choice
        """
        import os
        import sys

        # Return default without prompting in test/non-interactive environments or when auto_confirm is True
        if (
            auto_confirm
            or "pytest" in sys.modules
            or "PYTEST_CURRENT_TEST" in os.environ
            or not sys.stdin.isatty()  # Non-interactive environment
        ):
            return default

        import typer

        return typer.confirm(message, default=default)

    def show_message(self, message: str) -> None:
        """Display message to user using Rich.

        Args:
            message: Message to display
        """
        from rich.console import Console

        console = Console()
        # Check if message contains rich markup
        if any(marker in message for marker in ["[bold", "[cyan", "[yellow", "[red"]):
            console.print(message)
        else:
            console.print(message)


class StandardFileSystemOperations(FileSystemOperations):
    """Standard file system operations (SRP: Only handles file I/O)."""

    def create_directory(self, path: Path) -> None:
        """Create directory if it doesn't exist.

        Args:
            path: Directory path to create
        """
        path.mkdir(parents=True, exist_ok=True)

    def directory_exists(self, path: Path) -> bool:
        """Check if directory exists.

        Args:
            path: Directory path to check

        Returns:
            True if directory exists
        """
        return path.exists() and path.is_dir()


class MLflowContextFactory(ServerContextFactory):
    """Factory for MLflow server contexts (SRP: Only creates contexts)."""

    def create_server_context(self, mlflow_settings: Any, **overrides: Any) -> Any:
        """Create MLflow server context from settings.

        Args:
            mlflow_settings: MLflow configuration settings
            **overrides: Configuration overrides

        Returns:
            MLflowServerContext ready for use
        """
        from .server_configuration import validate_mlflow_config
        from .mlflow_adapter import MLflowServerContext

        validate_mlflow_config(mlflow_settings)
        return MLflowServerContext(mlflow_settings.server, **overrides)
