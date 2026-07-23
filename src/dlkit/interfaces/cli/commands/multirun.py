"""Multirun sweep commands for DLKit CLI.

`dlkit multirun` is a command group with two subcommands:
- `dlkit multirun validate CONFIG.toml` — expand child sources without executing
  anything (no MLflow run opened), so sweep-definition mistakes (duplicate ids,
  bad globs, nested multirun children) surface before any training starts.
- `dlkit multirun run CONFIG.toml` — actually execute the sweep.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dlkit.engine.workflows.entrypoints import build_child_sources, expand_child_sources
from dlkit.infrastructure.config.job_config import MultiRunJobConfig
from dlkit.interfaces.api.functions.core import run_multirun_config as api_run_multirun_config

from ..adapters.config_adapter import load_config
from ..adapters.result_presenter import present_multirun_result
from ..middleware.error_handler import handle_cli_errors
from ..params import CONFIG_PATH_ARG, MLFLOW_FLAG

app = typer.Typer(
    name="multirun",
    help="🧪 Multirun sweeps — batch-run a set of child configs under one parent MLflow run",
    no_args_is_help=True,
)

console = Console()


def _load_multirun_config(config_path: Path) -> MultiRunJobConfig:
    """Load and type-narrow a multirun TOML config.

    Args:
        config_path: Path to the multirun TOML configuration file.

    Returns:
        Validated MultiRunJobConfig.
    """
    console.print(f"📖 Loading configuration from: {config_path}")
    return load_config(config_path, run_type="multirun")


@app.command("run")
@handle_cli_errors(console)
def run_sweep(
    config_path: CONFIG_PATH_ARG,
    mlflow: MLFLOW_FLAG = False,
) -> None:
    """Execute a multirun sweep.

    Examples:
        dlkit multirun run sweep.toml
    """
    job = _load_multirun_config(config_path)
    console.print(
        f"🧪 Starting multirun sweep '{job.parent_run_name}' ({len(job.runs)} child source(s))..."
    )
    result = api_run_multirun_config(job, mlflow=mlflow)
    console.print("🎉 Multirun sweep completed!")
    present_multirun_result(result, console)


@app.command("validate")
@handle_cli_errors(console)
def validate_sweep(config_path: CONFIG_PATH_ARG) -> None:
    """Expand a multirun sweep's children without executing anything.

    Loads the config, converts each `[[multirun.runs]]` entry into a
    ChildSource, and expands them into concrete RunSpecs — surfacing
    duplicate ids, zero-match globs, and nested multirun children before any
    MLflow run is opened.

    Examples:
        dlkit multirun validate sweep.toml
    """
    job = _load_multirun_config(config_path)
    sources = build_child_sources(job.runs)
    children = expand_child_sources(sources)

    console.print(f"✅ {len(children)} child run(s) expanded successfully:\n")
    table = Table(title="Multirun Children (dry run)")
    table.add_column("id", style="cyan")
    table.add_column("label", style="green")
    table.add_column("run_name", style="yellow")

    for child in children:
        table.add_row(child.id, child.label, child.run_name)

    console.print(table)
