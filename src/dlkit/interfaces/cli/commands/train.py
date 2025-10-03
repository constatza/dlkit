"""Training commands for DLKit CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from dlkit.interfaces.api import train as api_train, validate_config

from ..adapters.config_adapter import load_config
from ..adapters.result_presenter import present_training_result
from ..middleware.error_handler import handle_api_error

# Create training command group
app = typer.Typer(
    name="train",
    help="🏋️ Training commands — Train machine learning models",
)

console = Console()


def _run_training_impl(
    config_path: Annotated[
        Path,
        typer.Argument(
            help="Path to configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    mlflow: Annotated[bool, typer.Option("--mlflow", help="Enable MLflow tracking")] = False,
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", "-c", help="Path to checkpoint for resuming training"),
    ] = None,
    validate_only: Annotated[
        bool, typer.Option("--validate-only", help="Only validate configuration without training")
    ] = False,
    # Root override
    root_dir: Annotated[
        Path | None,
        typer.Option("--root-dir", help="Root directory for path resolution (overrides config)"),
    ] = None,
    # Basic overrides
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Override output directory from config"),
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option("--dataflow-dir", "-d", help="Override dataflow directory from config"),
    ] = None,
    # Training overrides
    epochs: Annotated[
        int | None, typer.Option("--epochs", "-e", help="Override number of training epochs")
    ] = None,
    batch_size: Annotated[
        int | None, typer.Option("--batch-size", "-b", help="Override batch size for training")
    ] = None,
    learning_rate: Annotated[
        float | None, typer.Option("--learning-rate", "-l", help="Override learning rate")
    ] = None,
    # MLflow overrides
    mlflow_host: Annotated[
        str | None, typer.Option("--mlflow-host", help="Override MLflow server hostname")
    ] = None,
    mlflow_port: Annotated[
        int | None, typer.Option("--mlflow-port", help="Override MLflow server port")
    ] = None,
    experiment_name: Annotated[
        str | None, typer.Option("--experiment-name", help="Override MLflow experiment name")
    ] = None,
    run_name: Annotated[
        str | None, typer.Option("--run-name", help="Override MLflow run name")
    ] = None,
) -> None:
    """Run training workflow with configuration and parameter overrides.

    Examples:
        dlkit train config.toml
        dlkit train config.toml --mlflow --epochs 100
        dlkit train config.toml --checkpoint model.ckpt --output-dir ./custom
        dlkit train config.toml --mlflow --mlflow-host localhost --experiment-name test
        dlkit train config.toml --validate-only
    """
    try:
        # Load configuration (don't apply output_dir here, let API handle all overrides)
        console.print(f"📖 Loading configuration from: {config_path}")
        try:
            settings = load_config(config_path, root_dir=root_dir, workflow_type="training")
        except Exception as e:
            handle_api_error(e, console)
            raise typer.Exit(1)

        # Show training mode
        if mlflow or (settings.MLFLOW and settings.MLFLOW.enabled):
            console.print("🎯 Using [bold]training with MLflow tracking[/bold]")
        else:
            console.print("🎯 Using [bold]vanilla training[/bold]")

        # Validate configuration
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Validating configuration...", total=None)
            validate_config(settings)

        console.print("✅ Configuration validated successfully")

        if validate_only:
            console.print("🏁 Validation complete (--validate-only specified)")
            return

        # Execute training with overrides
        if checkpoint:
            console.print(f"🔄 Resuming training from checkpoint: {checkpoint}")
        else:
            console.print("🚀 Starting training workflow...")

        # Show applied overrides
        override_messages = []
        if checkpoint:
            override_messages.append(f"Checkpoint: {checkpoint}")
        if output_dir:
            override_messages.append(f"Output dir: {output_dir}")
        if data_dir:
            override_messages.append(f"Data dir: {data_dir}")
        if epochs:
            override_messages.append(f"Epochs: {epochs}")
        if batch_size:
            override_messages.append(f"Batch size: {batch_size}")
        if learning_rate:
            override_messages.append(f"Learning rate: {learning_rate}")
        if mlflow_host:
            override_messages.append(f"MLflow host: {mlflow_host}")
        if mlflow_port:
            override_messages.append(f"MLflow port: {mlflow_port}")
        if experiment_name:
            override_messages.append(f"Experiment: {experiment_name}")
        if run_name:
            override_messages.append(f"Run name: {run_name}")

        if root_dir:
            override_messages.append(f"Root dir: {root_dir}")
        if override_messages:
            console.print("🔧 Parameter overrides:")
            for msg in override_messages:
                console.print(f"  • {msg}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Training in progress...", total=None)
            training_result = api_train(
                settings,
                mlflow=mlflow,
                checkpoint_path=checkpoint,
                root_dir=root_dir,
                output_dir=output_dir,
                data_dir=data_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                mlflow_host=mlflow_host,
                mlflow_port=mlflow_port,
                experiment_name=experiment_name,
                run_name=run_name,
            )
            progress.remove_task(task)

        result = training_result
        console.print("🎉 Training completed successfully!")
        present_training_result(result, console)

    except typer.Exit:
        raise
    except Exception as e:
        # Handle DLKit errors (training failures, validation errors, etc.)
        from dlkit.interfaces.api.domain.errors import DLKitError

        if isinstance(e, DLKitError):
            handle_api_error(e, console)
        else:
            console.print(f"[red]Unexpected error during training: {e}[/red]")
        raise typer.Exit(1)


# Subcommands removed - functionality moved to main callback with flags:
# - Resume: use --checkpoint flag
# - Validate: use --validate-only flag


# Default training entry (no subcommands): dlkit train <config>
@app.callback(invoke_without_command=True)
def entry(
    config_path: Annotated[
        Path,
        typer.Argument(
            help="Path to configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    mlflow: Annotated[bool, typer.Option("--mlflow", help="Enable MLflow tracking")] = False,
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", "-c", help="Path to checkpoint for resuming training"),
    ] = None,
    validate_only: Annotated[
        bool, typer.Option("--validate-only", help="Only validate configuration without training")
    ] = False,
    # Root override
    root_dir: Annotated[
        Path | None,
        typer.Option("--root-dir", help="Root directory for path resolution (overrides config)"),
    ] = None,
    # Basic overrides
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Override output directory from config"),
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option("--dataflow-dir", "-d", help="Override dataflow directory from config"),
    ] = None,
    # Training overrides
    epochs: Annotated[
        int | None, typer.Option("--epochs", "-e", help="Override number of training epochs")
    ] = None,
    batch_size: Annotated[
        int | None, typer.Option("--batch-size", "-b", help="Override batch size for training")
    ] = None,
    learning_rate: Annotated[
        float | None, typer.Option("--learning-rate", "-l", help="Override learning rate")
    ] = None,
    # MLflow overrides
    mlflow_host: Annotated[
        str | None, typer.Option("--mlflow-host", help="Override MLflow server hostname")
    ] = None,
    mlflow_port: Annotated[
        int | None, typer.Option("--mlflow-port", help="Override MLflow server port")
    ] = None,
    experiment_name: Annotated[
        str | None, typer.Option("--experiment-name", help="Override MLflow experiment name")
    ] = None,
    run_name: Annotated[
        str | None, typer.Option("--run-name", help="Override MLflow run name")
    ] = None,
) -> None:
    """Train machine learning models with configuration and overrides.

    Usage:
      dlkit train CONFIG.toml [options]
    """
    _run_training_impl(
        config_path=config_path,
        mlflow=mlflow,
        checkpoint=checkpoint,
        validate_only=validate_only,
        root_dir=root_dir,
        output_dir=output_dir,
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mlflow_host=mlflow_host,
        mlflow_port=mlflow_port,
        experiment_name=experiment_name,
        run_name=run_name,
    )
