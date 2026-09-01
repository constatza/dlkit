"""Error handling middleware for CLI."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar, cast

import pydantic
import typer
from pydantic_core import ErrorDetails
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from dlkit.common import (
    ConfigurationError,
    DLKitError,
    ModelStateError,
    PluginError,
    StrategyError,
    TrackingError,
    WorkflowError,
)

F = TypeVar("F", bound=Callable[..., Any])


def _configuration_error_suggestions(error: DLKitError) -> list[str]:
    """Build suggestions for configuration errors.

    Args:
        error: Configuration error instance.

    Returns:
        List of suggestion strings.
    """
    suggestions = [
        "Check your configuration file syntax and formatting",
        "Validate configuration: dlkit config validate <config_file>",
        "Create a template: dlkit config create --output config.toml",
    ]
    if config_path := error.context.get("config_path"):
        suggestions.append(f"Verify file exists: {config_path}")
    return suggestions


def _strategy_error_suggestions(error: DLKitError) -> list[str]:
    """Build suggestions for strategy errors.

    Args:
        error: Strategy error instance.

    Returns:
        List of suggestion strings.
    """
    suggestions = [
        "Verify the strategy name is correct (training, mlflow, optuna, inference)",
        "Check that required plugins are enabled in configuration",
        "Validate strategy compatibility: dlkit config validate <config> --strategy <strategy>",
    ]
    if "available_modes" in error.context:
        modes = ", ".join(error.context["available_modes"])
        suggestions.append(f"Available strategies: {modes}")
    return suggestions


def _plugin_error_suggestions(error: DLKitError) -> list[str]:
    """Build suggestions for plugin errors, naming the offending plugin.

    Args:
        error: Plugin error instance.

    Returns:
        List of suggestion strings with the plugin name substituted in.
    """
    plugin_name = error.context.get("plugin", "unknown")
    defaults = [
        "Enable the required plugin in configuration",
        "Check plugin configuration parameters",
        "Verify plugin dependencies are installed",
    ]
    return [
        suggestion.replace("the required plugin", plugin_name).replace("plugin", plugin_name)
        if "plugin" in suggestion
        else suggestion
        for suggestion in defaults
    ]


def _model_state_error_suggestions(error: DLKitError) -> list[str]:
    """Build suggestions for model state errors.

    Args:
        error: Model state error instance.

    Returns:
        List of suggestion strings.
    """
    return [
        "Check model configuration and parameters",
        "Verify dataflow module configuration",
        "Ensure trainer settings are compatible",
    ]


def _workflow_error_suggestions(error: DLKitError) -> list[str]:
    """Build suggestions for workflow errors, adding strategy-specific hints.

    Args:
        error: Workflow error instance.

    Returns:
        List of suggestion strings.
    """
    suggestions = [
        "Check log files for detailed error information",
        "Verify system resources (memory, GPU, disk space)",
        "Try running with --verbose for more details",
    ]
    match error.context.get("strategy"):
        case "mlflow":
            suggestions.append("Check MLflow tracking URI/env configuration and connectivity")
        case "optuna":
            suggestions.append("Verify Optuna study configuration and storage")
    return suggestions


def _tracking_error_suggestions(error: DLKitError) -> list[str]:
    """Build suggestions for tracking backend errors.

    Args:
        error: Tracking error instance.

    Returns:
        List of suggestion strings.
    """
    return [
        "Check the MLflow tracking URI is reachable (VPN/firewall/server up)",
        "Set tracking.on_connectivity_failure = 'fallback_local' to track "
        "locally instead of failing when the backend is unreachable",
    ]


_ERROR_SUGGESTIONS: dict[type[DLKitError], Callable[[DLKitError], list[str]]] = {
    ConfigurationError: _configuration_error_suggestions,
    StrategyError: _strategy_error_suggestions,
    PluginError: _plugin_error_suggestions,
    ModelStateError: _model_state_error_suggestions,
    WorkflowError: _workflow_error_suggestions,
    TrackingError: _tracking_error_suggestions,
}


def handle_api_error(error: DLKitError, console: Console) -> None:
    """Handle API errors with appropriate CLI formatting.

    Args:
        error: DLKit domain error
        console: Rich console for output
    """
    # Create error text with appropriate styling
    error_text = Text()

    # Add error type and message
    error_type = type(error).__name__
    error_text.append(f"{error_type}\n", style="bold red")
    error_text.append(f"{error.message}\n", style="red")

    # Add context information if available
    if error.context:
        error_text.append("\nContext:\n", style="yellow")
        for key, value in error.context.items():
            error_text.append(f"  {key}: {value}\n", style="dim")

    # Add helpful suggestions based on error type
    suggestions = _get_error_suggestions(error)
    if suggestions:
        error_text.append("\n💡 Suggestions:\n", style="blue")
        for suggestion in suggestions:
            error_text.append(f"  • {suggestion}\n", style="cyan")

    # Create error panel
    error_panel = Panel.fit(error_text, title="❌ Error", border_style="red")

    console.print(error_panel)


def _get_error_suggestions(error: DLKitError) -> list[str]:
    """Get helpful suggestions based on error type.

    Args:
        error: DLKit domain error

    Returns:
        List of suggestion strings
    """
    for error_type, build_suggestions in _ERROR_SUGGESTIONS.items():
        if isinstance(error, error_type):
            return build_suggestions(error)
    return []


def _describe_pydantic_error(detail: ErrorDetails) -> str:
    """Translate a single structured pydantic error into a user-friendly message.

    Args:
        detail: One entry from ``pydantic.ValidationError.errors()``.

    Returns:
        A short, user-facing description of the error.
    """
    error_type = detail["type"]
    match error_type:
        case "missing":
            return "Required field is missing"
        case "value_error":
            return "Invalid value"
        case _ if error_type.endswith(("_type", "_parsing")):
            return "Invalid dataflow type"
        case _:
            return detail["msg"]


def format_validation_error(error: Exception) -> str:
    """Format validation errors in a user-friendly way.

    Args:
        error: Validation exception

    Returns:
        Formatted error message
    """
    if not isinstance(error, pydantic.ValidationError):
        return str(error)

    return "\n".join(f"• {_describe_pydantic_error(detail)}" for detail in error.errors())


def handle_keyboard_interrupt(console: Console) -> None:
    """Handle keyboard interrupt gracefully.

    Args:
        console: Rich console for output
    """
    interrupt_text = Text()
    interrupt_text.append("Operation cancelled by user", style="yellow")

    interrupt_panel = Panel.fit(interrupt_text, title="⚠️ Interrupted", border_style="yellow")

    console.print(interrupt_panel)


def handle_unexpected_error(error: Exception, console: Console) -> None:
    """Handle unexpected errors with debugging information.

    Args:
        error: Unexpected exception
        console: Rich console for output
    """
    error_text = Text()
    error_text.append("Unexpected Error\n", style="bold red")
    error_text.append(f"{type(error).__name__}: {error}\n", style="red")

    # Add debugging suggestions
    error_text.append("\n💡 Debug Steps:\n", style="blue")
    error_text.append("  • Run with --verbose for detailed output\n", style="cyan")
    error_text.append("  • Check log files in output directory\n", style="cyan")
    error_text.append("  • Validate configuration: dlkit config validate <config>\n", style="cyan")
    error_text.append("  • Report issue with full error trace if problem persists\n", style="cyan")

    error_panel = Panel.fit(error_text, title="🐛 Unexpected Error", border_style="red")

    console.print(error_panel)


def handle_cli_errors(console: Console) -> Callable[[F], F]:
    """Decorator that wraps CLI commands to handle errors consistently.

    Catches DLKitError and unexpected exceptions, formats them, and exits cleanly.

    Args:
        console: Rich console for output

    Returns:
        Decorator function
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except typer.Exit:
                # Re-raise typer.Exit without modification
                raise
            except DLKitError as e:
                # Handle domain errors with formatted suggestions
                handle_api_error(e, console)
                raise typer.Exit(1)
            except Exception as e:
                # Handle unexpected errors with debug information
                handle_unexpected_error(e, console)
                raise typer.Exit(1)

        return cast(F, wrapper)

    return decorator
