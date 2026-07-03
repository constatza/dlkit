"""Convergence study commands for DLKit CLI."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from dlkit.common.results import ConvergenceResult
from dlkit.engine.workflows.entrypoints._override_types import ConvergenceOverrides
from dlkit.interfaces.api.functions.core import converge as api_converge

from ..adapters.config_adapter import load_config
from ..middleware.error_handler import handle_cli_errors
from ..params import (
    CONFIG_PATH_ARG,
    EXPERIMENT_NAME_PARAM,
    MLFLOW_FLAG,
    RUN_NAME_PARAM,
)

app = typer.Typer(
    name="converge",
    help="📈 Convergence study — analyse model performance vs. training-set size",
)

console = Console()


def _parse_sizes(sizes_str: str) -> tuple[int, ...]:
    """Parse a comma-separated string of integers into a tuple.

    Args:
        sizes_str: Comma-separated integers, e.g. ``"100,500,1000"``.

    Returns:
        Tuple of parsed positive integers.

    Raises:
        typer.BadParameter: If any value is not a valid positive integer.
    """
    parts = [s.strip() for s in sizes_str.split(",") if s.strip()]
    try:
        values = tuple(int(p) for p in parts)
    except ValueError:
        raise typer.BadParameter(f"--sizes must be comma-separated integers, got: {sizes_str!r}")
    for v in values:
        if v <= 0:
            raise typer.BadParameter(f"All sizes must be positive integers; got {v}")
    return values


def _present_convergence_result(result: ConvergenceResult) -> None:
    """Print a formatted summary table for a convergence result.

    Args:
        result: ConvergenceResult to present.
    """
    console.print(f"\nDuration: {result.duration_seconds:.2f}s")
    if result.n_star is not None:
        console.print(f"n* (converged at): {result.n_star}")
    else:
        console.print("n*: not reached (no target set or threshold not met)")

    if result.mlflow_run_id:
        console.print(f"MLflow run: {result.mlflow_run_id}")

    table = Table(title="Convergence Points")
    table.add_column("n", style="cyan")
    table.add_column("val_mean", style="green")
    table.add_column("val_std", style="yellow")
    table.add_column("gap", style="magenta")
    table.add_column("marginal_gain", style="blue")
    table.add_column("converged", style="bold")

    for pt in result.points:
        gain_str = f"{pt.marginal_gain:.6f}" if pt.marginal_gain is not None else "—"
        conv_str = "✅" if pt.converged is True else ("❌" if pt.converged is False else "—")
        table.add_row(
            str(pt.n),
            f"{pt.val_mean:.6f}",
            f"{pt.val_std:.6f}",
            f"{pt.gap:.6f}",
            gain_str,
            conv_str,
        )

    console.print(table)


@handle_cli_errors(console)
def _run_convergence_impl(
    config_path: CONFIG_PATH_ARG,
    mlflow: MLFLOW_FLAG = False,
    sizes: str | None = typer.Option(
        None,
        "--sizes",
        help="Comma-separated training-set sizes, e.g. '100,500,1000'",
    ),
    repeats: int | None = typer.Option(None, "--repeats", help="Independent runs per size"),
    target: float | None = typer.Option(None, "--target", help="Convergence threshold (E_target)"),
    experiment_name: EXPERIMENT_NAME_PARAM = None,
    run_name: RUN_NAME_PARAM = None,
) -> None:
    """Run a sample-size convergence study.

    Examples:
        dlkit converge config.toml
        dlkit converge config.toml --sizes 100,500,1000 --repeats 3
        dlkit converge config.toml --target 0.05 --mlflow
    """
    console.print(f"Loading configuration from: {config_path}")
    job = load_config(config_path, run_type="converge")

    console.print("Starting convergence study...")

    parsed_sizes: tuple[int, ...] | None = None
    if sizes is not None:
        parsed_sizes = _parse_sizes(sizes)
        console.print(f"Override sizes: {parsed_sizes}")

    overrides = ConvergenceOverrides(
        sizes=parsed_sizes,
        repeats=repeats,
        target=target,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Convergence study in progress...", total=None)
        result: ConvergenceResult = api_converge(job, overrides=overrides, mlflow=mlflow)
        progress.remove_task(task)

    console.print("Convergence study completed!")
    _present_convergence_result(result)


@app.callback(invoke_without_command=True)
def entry(
    config_path: CONFIG_PATH_ARG,
    mlflow: MLFLOW_FLAG = False,
    sizes: str | None = typer.Option(
        None,
        "--sizes",
        help="Comma-separated training-set sizes, e.g. '100,500,1000'",
    ),
    repeats: int | None = typer.Option(None, "--repeats", help="Independent runs per size"),
    target: float | None = typer.Option(None, "--target", help="Convergence threshold (E_target)"),
    experiment_name: EXPERIMENT_NAME_PARAM = None,
    run_name: RUN_NAME_PARAM = None,
) -> None:
    """Run a sample-size convergence study.

    Usage:
      dlkit converge CONFIG.toml [options]
    """
    _run_convergence_impl(
        config_path=config_path,
        mlflow=mlflow,
        sizes=sizes,
        repeats=repeats,
        target=target,
        experiment_name=experiment_name,
        run_name=run_name,
    )
