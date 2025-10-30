"""Shared CLI parameter definitions for DLKit commands.

This module provides reusable Typer parameter definitions using `Annotated` types
to eliminate duplication across CLI command files. All parameters are defined as
constants that can be imported and used directly in command signatures.

Design rationale:
- DRY principle: Define each parameter once, use everywhere
- Consistency: All commands use identical parameter definitions
- Maintainability: Changes to parameter help text or types happen in one place
- Type safety: Full type hints maintained via Annotated types

Usage example:
    from dlkit.interfaces.cli.params import ROOT_DIR_PARAM, OUTPUT_DIR_PARAM

    @app.command()
    def my_command(
        root_dir: ROOT_DIR_PARAM = None,
        output_dir: OUTPUT_DIR_PARAM = None,
    ) -> None:
        # Command implementation
        pass

Note:
- All path parameters use `Path | None` (not `Optional[Path]`)
- Parameter names in function signatures must match the annotated name
- CLI shortcuts (like `-o` for `--output-dir`) are preserved in annotations
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


# ============================================================================
# Path Override Parameters
# ============================================================================

ROOT_DIR_PARAM = Annotated[
    Path | None,
    typer.Option("--root-dir", help="Root directory for path resolution (overrides config)"),
]
"""Root directory override for session-level path resolution.

Overrides the root directory specified in config's SESSION.root_dir or
the DLKIT_ROOT_DIR environment variable. All relative paths in the
configuration are resolved relative to this root.
"""

OUTPUT_DIR_PARAM = Annotated[
    Path | None,
    typer.Option("--output-dir", "-o", help="Override output directory from config"),
]
"""Output directory override for predictions and results.

Overrides the output directory where predictions, metrics, and other
training artifacts are saved. Defaults to `output/` under the root.
"""

DATA_DIR_PARAM = Annotated[
    Path | None,
    typer.Option("--dataflow-dir", "-d", help="Override dataflow directory from config"),
]
"""Data directory override for input datasets.

Overrides the directory where input data files are located. This is
typically used to point to a different data location without modifying
the configuration file.
"""

CHECKPOINT_PARAM = Annotated[
    Path | None,
    typer.Option("--checkpoint", "-c", help="Path to checkpoint for resuming training"),
]
"""Checkpoint path for resuming training or loading trained models.

Points to a PyTorch Lightning checkpoint file (.ckpt) to resume training
from a previous run or to load a trained model for inference.
"""

CHECKPOINT_ARG = Annotated[
    Path,
    typer.Argument(help="Path to model checkpoint"),
]
"""Required checkpoint argument (non-optional) for inference commands.

Used when the checkpoint is a required positional argument rather than
an optional flag.
"""


# ============================================================================
# Training Hyperparameter Overrides
# ============================================================================

EPOCHS_PARAM = Annotated[
    int | None,
    typer.Option("--epochs", "-e", help="Override number of training epochs"),
]
"""Training epochs override.

Overrides the number of training epochs specified in the configuration.
Must be a positive integer.
"""

BATCH_SIZE_PARAM = Annotated[
    int | None,
    typer.Option("--batch-size", "-b", help="Override batch size for training"),
]
"""Batch size override for training and inference.

Overrides the batch size used for DataLoader construction. Applies to
both training and inference workflows.
"""

LEARNING_RATE_PARAM = Annotated[
    float | None,
    typer.Option("--learning-rate", "-l", help="Override learning rate"),
]
"""Learning rate override for optimizer.

Overrides the learning rate specified in the optimizer configuration.
Useful for quick learning rate experiments without changing config files.
"""


# ============================================================================
# MLflow Server/Client Overrides
# ============================================================================

MLFLOW_FLAG = Annotated[
    bool,
    typer.Option("--mlflow", help="Enable MLflow tracking"),
]
"""MLflow tracking enablement flag.

Enables MLflow experiment tracking for training or optimization runs.
When enabled, metrics, parameters, and artifacts are logged to MLflow.
"""

MLFLOW_HOST_PARAM = Annotated[
    str | None,
    typer.Option("--mlflow-host", help="Override MLflow server hostname"),
]
"""MLflow server hostname override.

Overrides the hostname of the MLflow tracking server. Defaults to
'localhost' if not specified in config or via this parameter.
"""

MLFLOW_PORT_PARAM = Annotated[
    int | None,
    typer.Option("--mlflow-port", help="Override MLflow server port"),
]
"""MLflow server port override.

Overrides the port of the MLflow tracking server. Defaults to 5000
if not specified in config or via this parameter.
"""

EXPERIMENT_NAME_PARAM = Annotated[
    str | None,
    typer.Option("--experiment-name", help="Override MLflow experiment name"),
]
"""MLflow experiment name override.

Overrides the experiment name used for organizing runs in MLflow.
Experiments group related runs together in the MLflow UI.
"""

RUN_NAME_PARAM = Annotated[
    str | None,
    typer.Option("--run-name", help="Override MLflow run name"),
]
"""MLflow run name override.

Overrides the name assigned to this specific training or optimization run.
Run names appear in the MLflow UI and help identify individual experiments.
"""


# ============================================================================
# Configuration and Validation Parameters
# ============================================================================

CONFIG_PATH_ARG = Annotated[
    Path,
    typer.Argument(
        help="Path to configuration file",
        # Note: exists=False allows application code to handle missing files with better error messages
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
]
"""Required configuration file path argument.

Points to a TOML configuration file that defines the training pipeline,
model architecture, datasets, and other settings. The file must exist
and be readable.
"""

VALIDATE_ONLY_FLAG = Annotated[
    bool,
    typer.Option("--validate-only", help="Only validate configuration without training"),
]
"""Validation-only mode flag.

When enabled, the configuration is loaded and validated but no training
is performed. Useful for checking configuration correctness before
starting expensive training runs.
"""


# ============================================================================
# Inference-Specific Parameters
# ============================================================================

SAVE_PREDICTIONS_FLAG = Annotated[
    bool,
    typer.Option("--save", "-s", help="Save predictions to file"),
]
"""Save predictions flag for inference.

When enabled, prediction results are saved to disk in the output directory.
The output format depends on the model and data type (arrays, graphs, etc.).
"""
