"""Main Typer application for DLKit CLI."""

from __future__ import annotations

import sys
import os

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from dlkit.tools.utils.logging_config import configure_logging, get_logger
from .commands import optimize, config, convert
from .commands import train as train
from .commands import predict as predict

logger = get_logger(__name__)

# Create main Typer application
app = typer.Typer(
    name="dlkit",
    help="🧠 Deep Learning Toolkit - Train, optimize, and infer with ML models",
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
)

# Add top-level sub-apps for train/predict (expose their help/structure)
app.add_typer(train.app, name="train", help="🏋️ Training commands — Train machine learning models")
app.add_typer(
    predict.app,
    name="predict",
    help="🔮 Prediction — Run predictions with trained models using training configs",
)

# Keep other command groups
app.add_typer(
    convert.app, name="convert", help="🔁 Convert checkpoints to export formats (e.g., ONNX)"
)
app.add_typer(optimize.app, name="optimize", help="⚡ Hyperparameter optimization commands")
app.add_typer(config.app, name="config", help="⚙️ Configuration validation and utilities")

console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version information"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Set logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
) -> None:
    """DLKit - Deep Learning Toolkit with modern architecture.

    A comprehensive toolkit for training, optimizing, and running inference
    with machine learning models using Lightning, MLflow, and Optuna.
    """
    # Configure logging first, before other logic
    debug_enabled = debug or verbose
    log_level_final = "DEBUG" if debug_enabled else log_level.upper()

    try:
        configure_logging(
            level=log_level_final,
            debug_enabled=debug_enabled,
            format_type="simple" if not debug_enabled else "structured",
        )
        logger.debug(
            "DLKit CLI initialized", debug_enabled=debug_enabled, log_level=log_level_final
        )
    except Exception as e:
        # Fallback if logging configuration fails
        print(f"Warning: Logging configuration failed: {e}")
        pass

    if version:
        show_version()
        raise typer.Exit()

    # If no subcommand was invoked and no special flags, show help
    if ctx.invoked_subcommand is None and not version:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def show_version() -> None:
    """Display version information."""
    from dlkit import __version__

    version_text = __version__

    version_panel = Panel.fit(
        Text(f"DLKit v{version_text}", style="bold green"), title="Version", border_style="green"
    )
    console.print(version_panel)


def _get_version(module_name: str, module) -> str:
    """Safely get version from a module."""
    try:
        if hasattr(module, "__version__"):
            return module.__version__
        elif hasattr(module, "version"):
            return module.version
        elif module_name == "mlflow":
            return getattr(module, "version", {}).get("VERSION", "unknown")
        else:
            return "unknown"
    except Exception:
        return "unknown"


@app.command("info")
def show_info() -> None:
    """Show system and DLKit information."""
    try:
        import torch
        import lightning
        import mlflow
        import optuna

        info_text = Text()
        info_text.append("🧠 DLKit - Deep Learning Toolkit\n\n", style="bold blue")
        info_text.append("Dependencies:\n", style="bold")
        info_text.append(f"  • PyTorch: {_get_version('torch', torch)}\n")
        info_text.append(f"  • Lightning: {_get_version('lightning', lightning)}\n")
        info_text.append(f"  • MLflow: {_get_version('mlflow', mlflow)}\n")
        info_text.append(f"  • Optuna: {_get_version('optuna', optuna)}\n")

        # Add Python and system info
        info_text.append(
            f"\nPython: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n"
        )
        info_text.append(f"Platform: {sys.platform}\n")

        info_panel = Panel.fit(info_text, title="System Information", border_style="blue")
        console.print(info_panel)

    except Exception as e:
        error_text = Text(f"Unexpected error getting system info: {e}", style="bold red")
        error_panel = Panel.fit(error_text, title="Error", border_style="red")
        console.print(error_panel)


def cli_main() -> None:
    """Entry point for CLI application."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    cli_main()
