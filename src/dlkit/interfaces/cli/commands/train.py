"""Training commands for DLKit CLI."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from dlkit.interfaces.api import train as api_train
from dlkit.interfaces.api import validate_config

from ..adapters.config_adapter import load_config
from ..adapters.result_presenter import present_training_result
from ..guards import is_training_settings
from ..middleware.error_handler import handle_cli_errors
from ..params import (
    BATCH_SIZE_PARAM,
    CHECKPOINT_PARAM,
    CONFIG_PATH_ARG,
    DATA_DIR_PARAM,
    EPOCHS_PARAM,
    EXPERIMENT_NAME_PARAM,
    LEARNING_RATE_PARAM,
    MLFLOW_FLAG,
    OUTPUT_DIR_PARAM,
    ROOT_DIR_PARAM,
    RUN_NAME_PARAM,
    VALIDATE_ONLY_FLAG,
)

# Create training command group
app = typer.Typer(
    name="train",
    help="🏋️ Training commands — Train machine learning models",
)

console = Console()


@handle_cli_errors(console)
def _run_training_impl(
    config_path: CONFIG_PATH_ARG,
    mlflow: MLFLOW_FLAG = False,
    checkpoint: CHECKPOINT_PARAM = None,
    validate_only: VALIDATE_ONLY_FLAG = False,
    root_dir: ROOT_DIR_PARAM = None,
    output_dir: OUTPUT_DIR_PARAM = None,
    data_dir: DATA_DIR_PARAM = None,
    epochs: EPOCHS_PARAM = None,
    batch_size: BATCH_SIZE_PARAM = None,
    learning_rate: LEARNING_RATE_PARAM = None,
    experiment_name: EXPERIMENT_NAME_PARAM = None,
    run_name: RUN_NAME_PARAM = None,
) -> None:
    """Run training workflow with configuration and parameter overrides.

    Examples:
        dlkit train config.toml
        dlkit train config.toml --mlflow --epochs 100
        dlkit train config.toml --checkpoint model.ckpt --output-dir ./custom
        dlkit train config.toml --mlflow --experiment-name test
        dlkit train config.toml --validate-only
    """
    # Load configuration (don't apply output_dir here, let API handle all overrides)
    console.print(f"📖 Loading configuration from: {config_path}")
    settings = load_config(config_path, root_dir=root_dir)

    training_settings = settings if is_training_settings(settings) else None

    # Show training mode
    if mlflow or (training_settings and training_settings.MLFLOW):
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
            overrides={
                "checkpoint_path": checkpoint,
                "root_dir": root_dir,
                "output_dir": output_dir,
                "data_dir": data_dir,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "experiment_name": experiment_name,
                "run_name": run_name,
            },
            mlflow=mlflow,
        )
        progress.remove_task(task)

    result = training_result
    console.print("🎉 Training completed successfully!")
    present_training_result(result, console)


# Subcommands removed - functionality moved to main callback with flags:
# - Resume: use --checkpoint flag
# - Validate: use --validate-only flag


# Default training entry (no subcommands): dlkit train <config>
@app.callback(invoke_without_command=True)
def entry(
    config_path: CONFIG_PATH_ARG,
    mlflow: MLFLOW_FLAG = False,
    checkpoint: CHECKPOINT_PARAM = None,
    validate_only: VALIDATE_ONLY_FLAG = False,
    root_dir: ROOT_DIR_PARAM = None,
    output_dir: OUTPUT_DIR_PARAM = None,
    data_dir: DATA_DIR_PARAM = None,
    epochs: EPOCHS_PARAM = None,
    batch_size: BATCH_SIZE_PARAM = None,
    learning_rate: LEARNING_RATE_PARAM = None,
    experiment_name: EXPERIMENT_NAME_PARAM = None,
    run_name: RUN_NAME_PARAM = None,
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
        experiment_name=experiment_name,
        run_name=run_name,
    )
