"""Optimization commands for DLKit CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from dlkit.interfaces.api import optimize as api_optimize

from ..adapters.config_adapter import load_config
from ..adapters.result_presenter import present_optimization_result
from ..guards import is_training_settings
from ..middleware.error_handler import handle_cli_errors
from ..params import (
    CONFIG_PATH_ARG,
    MLFLOW_FLAG,
    OUTPUT_DIR_PARAM,
    ROOT_DIR_PARAM,
)

# Create optimization command group
app = typer.Typer(
    name="optimize",
    help="⚡ Hyperparameter optimization commands using Optuna",
    no_args_is_help=True,
)

console = Console()


@handle_cli_errors(console)
def _run_optimization_impl(
    config_path: CONFIG_PATH_ARG,
    trials: Annotated[
        int, typer.Option("--trials", "-n", help="Number of optimization trials")
    ] = 100,
    study_name: Annotated[
        str | None, typer.Option("--study-name", "-s", help="Name for the Optuna study")
    ] = None,
    mlflow: MLFLOW_FLAG = False,
    root_dir: ROOT_DIR_PARAM = None,
    output_dir: OUTPUT_DIR_PARAM = None,
) -> None:
    """Run hyperparameter optimization using Optuna.

    Examples:
        dlkit optimize config.toml --trials 50
        dlkit optimize config.toml --trials 100 --study-name my_study
        dlkit optimize config.toml --trials 50 --mlflow
    """
    # Load configuration
    console.print(f"📖 Loading configuration from: {config_path}")
    _settings = load_config(config_path, root_dir=root_dir)
    settings = _settings if is_training_settings(_settings) else None

    # Validate Optuna is configured (flattened)
    if not settings or not settings.OPTUNA or not settings.OPTUNA.enabled:
        console.print("[red]Optuna plugin must be enabled in configuration for optimization[/red]")
        console.print("Enable [OPTUNA] with enabled = true in config")
        raise typer.Exit(1)

    # Show optimization parameters
    console.print("⚡ Starting hyperparameter optimization")
    if mlflow or settings.MLFLOW:
        console.print("  With MLflow tracking enabled")
    console.print(f"  Trials: {trials}")
    if study_name:
        console.print(f"  Study name: {study_name}")
    if root_dir:
        console.print(f"  Root dir: {root_dir}")

    # Execute optimization
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Running {trials} optimization trials...", total=None)

        optimization_result = api_optimize(
            settings,
            overrides={
                "trials": trials,
                "study_name": study_name,
                "root_dir": root_dir,
                "output_dir": output_dir,
            },
            mlflow=mlflow,
        )
        progress.remove_task(task)

    result = optimization_result
    console.print("🎉 Optimization completed successfully!")
    present_optimization_result(result, console)


# Default optimization command: dlkit optimize config.toml --trials N
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config_path: CONFIG_PATH_ARG,
    trials: Annotated[
        int, typer.Option("--trials", "-n", help="Number of optimization trials")
    ] = 100,
    study_name: Annotated[
        str | None, typer.Option("--study-name", "-s", help="Name for the Optuna study")
    ] = None,
    mlflow: MLFLOW_FLAG = False,
    root_dir: ROOT_DIR_PARAM = None,
    output_dir: OUTPUT_DIR_PARAM = None,
) -> None:
    """Run hyperparameter optimization using Optuna with configuration and parameter overrides."""
    if ctx.invoked_subcommand is not None:
        return
    _run_optimization_impl(
        config_path=config_path,
        trials=trials,
        study_name=study_name,
        mlflow=mlflow,
        root_dir=root_dir,
        output_dir=output_dir,
    )


"""
Note: Removed superficial 'resume' subcommand. To add trials to an
existing study, pass the same `--study-name` to the main command.
"""


@app.command("status")
@handle_cli_errors(console)
def show_study_status(
    study_name: Annotated[str, typer.Argument(help="Name of the study")],
    storage: Annotated[str, typer.Argument(help="Storage URL for the study")],
) -> None:
    """Show status and progress of an Optuna study.

    Examples:
        dlkit optimize status my_study sqlite:///study.db
    """
    import optuna
    from rich.table import Table

    console.print(f"📊 Loading study status: {study_name}")

    # Load study
    study = optuna.load_study(study_name=study_name, storage=storage)

    # Create status table
    table = Table(title=f"Study Status: {study_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Study Name", study_name)
    table.add_row("Direction", study.direction.name)
    table.add_row("Total Trials", str(len(study.trials)))
    table.add_row(
        "Complete Trials", str(len([t for t in study.trials if t.state.name == "COMPLETE"]))
    )
    table.add_row("Failed Trials", str(len([t for t in study.trials if t.state.name == "FAIL"])))
    table.add_row("Pruned Trials", str(len([t for t in study.trials if t.state.name == "PRUNED"])))

    if study.best_trial:
        table.add_row("Best Value", f"{study.best_trial.value:.6f}")
        table.add_row("Best Trial", str(study.best_trial.number))

    console.print(table)

    # Show best parameters if available
    if study.best_trial:
        console.print("\n🏆 Best Parameters:")
        for param, value in study.best_trial.params.items():
            console.print(f"  {param}: {value}")


@app.command("plot")
@handle_cli_errors(console)
def plot_study(
    study_name: Annotated[str, typer.Argument(help="Name of the study")],
    storage: Annotated[str, typer.Argument(help="Storage URL for the study")],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Directory to save plots")
    ] = Path("plots"),
    plot_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Plot type: optimization_history, param_importances, parallel_coordinate",
        ),
    ] = "optimization_history",
) -> None:
    """Generate plots for an Optuna study.

    Examples:
        dlkit optimize plot my_study sqlite:///study.db --output-dir ./plots
        dlkit optimize plot my_study sqlite:///study.db --type param_importances
    """
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
        plot_slice,
    )

    console.print(f"📈 Generating plots for study: {study_name}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load study
    study = optuna.load_study(study_name=study_name, storage=storage)

    # Generate requested plot
    plot_functions = {
        "optimization_history": plot_optimization_history,
        "param_importances": plot_param_importances,
        "parallel_coordinate": plot_parallel_coordinate,
        "slice": plot_slice,
    }

    if plot_type not in plot_functions:
        console.print(f"[red]Unknown plot type: {plot_type}[/red]")
        console.print(f"Available types: {', '.join(plot_functions.keys())}")
        raise typer.Exit(1)

    plot_func = plot_functions[plot_type]
    fig = plot_func(study)

    # Save plot
    output_file = output_dir / f"{study_name}_{plot_type}.html"
    fig.write_html(str(output_file))

    console.print(f"✅ Plot saved to: {output_file}")
