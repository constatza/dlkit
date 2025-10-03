"""Error handling middleware for CLI."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from dlkit.interfaces.api.domain import (
    ConfigurationError,
    DLKitError,
    ModelStateError,
    PluginError,
    StrategyError,
    WorkflowError,
)


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
    suggestions = []

    if isinstance(error, ConfigurationError):
        suggestions.extend([
            "Check your configuration file syntax and formatting",
            "Validate configuration: dlkit config validate <config_file>",
            "Create a template: dlkit config create --output config.toml",
        ])

        # Specific suggestions based on context
        if error.context.get("config_path"):
            suggestions.append(f"Verify file exists: {error.context['config_path']}")

    elif isinstance(error, StrategyError):
        suggestions.extend([
            "Verify the strategy name is correct (training, mlflow, optuna, inference)",
            "Check that required plugins are enabled in configuration",
            "Validate strategy compatibility: dlkit config validate <config> --strategy <strategy>",
        ])

        if "available_modes" in error.context:
            modes = ", ".join(error.context["available_modes"])
            suggestions.append(f"Available strategies: {modes}")

    elif isinstance(error, PluginError):
        plugin_name = error.context.get("plugin", "unknown")
        suggestions.extend([
            f"Enable {plugin_name} plugin in configuration",
            f"Check {plugin_name} plugin configuration parameters",
            "Verify plugin dependencies are installed",
        ])

    elif isinstance(error, ModelStateError):
        suggestions.extend([
            "Check model configuration and parameters",
            "Verify dataflow module configuration",
            "Ensure trainer settings are compatible",
        ])

    elif isinstance(error, WorkflowError):
        suggestions.extend([
            "Check log files for detailed error information",
            "Verify system resources (memory, GPU, disk space)",
            "Try running with --verbose for more details",
        ])

        # Add strategy-specific suggestions
        strategy = error.context.get("strategy")
        if strategy == "mlflow":
            suggestions.append("Check MLflow server configuration and connectivity")
        elif strategy == "optuna":
            suggestions.append("Verify Optuna study configuration and storage")

    return suggestions


def format_validation_error(error: Exception) -> str:
    """Format validation errors in a user-friendly way.

    Args:
        error: Validation exception

    Returns:
        Formatted error message
    """
    error_msg = str(error)

    # Clean up common pydantic validation error patterns
    if "validation error" in error_msg.lower():
        # Extract field information from pydantic errors
        lines = error_msg.split("\n")
        simplified_lines = []

        for line in lines:
            if "field required" in line.lower():
                simplified_lines.append("• Required field is missing")
            elif "type_error" in line.lower():
                simplified_lines.append("• Invalid dataflow type")
            elif "value_error" in line.lower():
                simplified_lines.append("• Invalid value")
            else:
                simplified_lines.append(line)

        return "\n".join(simplified_lines)

    return error_msg


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
