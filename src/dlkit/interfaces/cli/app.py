"""Main Typer application for DLKit CLI."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import typer
from loguru import logger as loguru_logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from dlkit.infrastructure.config.environment import EnvironmentSettings
from dlkit.infrastructure.utils.logging_config import (
    configure_logging,
    get_effective_log_level,
    get_logger,
)

from .commands import config, convert, optimize
from .commands import converge as converge
from .commands import evaluate as evaluate
from .commands import evaluate_multirun as evaluate_multirun
from .commands import predict as predict
from .commands import train as train

logger = get_logger(__name__)


def _resolve_log_file_path() -> Path:
    """Resolve where `--log-file` should write to.

    Respects `DLKIT_LOG_FILE` if set; otherwise generates a fresh,
    timestamped path under `.dlkit/logs/`.
    """
    if env_override := os.getenv("DLKIT_LOG_FILE"):
        return Path(env_override)
    internal_dir = EnvironmentSettings().get_internal_dir_path()
    return internal_dir / "logs" / f"dlkit_{datetime.now():%Y%m%d_%H%M%S}.log"


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
app.add_typer(
    evaluate.app,
    name="evaluate",
    help="📊 Evaluation — Stats/plots for a trained checkpoint against labeled data",
)
app.add_typer(
    evaluate_multirun.app,
    name="evaluate-multirun",
    help="📊 Multirun evaluation — Batch-evaluate every child run of a sweep",
)

# Keep other command groups
app.add_typer(
    convert.app, name="convert", help="🔁 Convert checkpoints to export formats (e.g., ONNX)"
)
app.add_typer(optimize.app, name="optimize", help="⚡ Hyperparameter optimization commands")
app.add_typer(config.app, name="config", help="⚙️ Configuration validation and utilities")
app.add_typer(converge.app, name="converge", help="📈 Sample-size convergence study")

console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    log_level: str | None = typer.Option(
        None, "--log-level", help="Set logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
    log_file: bool = typer.Option(
        False,
        "--log-file",
        help="Also write logs to a timestamped file under .dlkit/logs/ "
        "(or the path in DLKIT_LOG_FILE)",
    ),
) -> None:
    """DLKit - Deep Learning Toolkit with modern architecture.

    A comprehensive toolkit for training, optimizing, and running inference
    with machine learning models using Lightning, MLflow, and Optuna.
    """
    # Configure logging first, before other logic
    debug_enabled = debug or verbose
    log_level_final = get_effective_log_level(level=log_level, debug_enabled=debug_enabled)
    log_file_path = _resolve_log_file_path() if log_file else None

    try:
        configure_logging(
            level=log_level,
            debug_enabled=debug_enabled,
            format_type="simple" if not debug_enabled else "structured",
            log_file=log_file_path,
        )
        logger.debug(
            "DLKit CLI initialized with level '{}' (debug_enabled={})",
            log_level_final,
            debug_enabled,
        )
    except Exception as e:
        loguru_logger.remove()
        loguru_logger.add(sys.stderr, level="WARNING")
        loguru_logger.warning("Logging configuration failed: {}", e)

    # If no subcommand was invoked and no special flags, show help
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command("info")
def show_info() -> None:
    """Show system and DLKit environment information."""
    try:
        import lightning
        import mlflow
        import optuna
        import torch

        info_text = Text()
        info_text.append("🧠 DLKit - Deep Learning Toolkit\n\n", style="bold blue")
        info_text.append("Dependencies:\n", style="bold")
        info_text.append(f"  • PyTorch: {torch.__name__}\n")
        info_text.append(f"  • Lightning: {lightning.__name__}\n")
        info_text.append(f"  • MLflow: {mlflow.__name__}\n")
        info_text.append(f"  • Optuna: {optuna.__name__}\n")

        info_text.append(f"\nPython executable: {sys.executable}\n")
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
