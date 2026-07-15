"""Evaluation command for DLKit CLI.

Evaluates a trained checkpoint against a labeled test/predict split: computes
MAE/RMSE/R2 and generates the same regression plots produced during training
— without running a training loop. Distinct from `dlkit predict`, which
returns raw predictions with no ground truth, metrics, or plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from rich.console import Console

from dlkit.infrastructure.config.job_config import InferenceJobConfig
from dlkit.interfaces.inference import evaluate as evaluate_api

from ..adapters.config_adapter import load_config
from ..adapters.result_presenter import present_evaluation_result
from ..middleware.error_handler import handle_cli_errors
from ..params import (
    BATCH_SIZE_PARAM,
    CHECKPOINT_ARG,
    CONFIG_PATH_ARG,
    MLFLOW_FLAG,
    RUN_NAME_PARAM,
)

app = typer.Typer(
    name="evaluate",
    help="📊 Evaluation — Stats/plots for a trained checkpoint against labeled data.",
)

console = Console()

SPLIT_PARAM = Annotated[
    str,
    typer.Option("--split", help="Labeled split to evaluate against: 'test' or 'predict'."),
]

OUTPUT_DIR_PARAM = Annotated[
    Path | None,
    typer.Option("--output-dir", help="Directory to save generated figures locally."),
]


@handle_cli_errors(console)
def _run_evaluate_impl(
    config_path: CONFIG_PATH_ARG,
    checkpoint: CHECKPOINT_ARG,
    split: SPLIT_PARAM = "test",
    batch_size: BATCH_SIZE_PARAM = None,
    output_dir: OUTPUT_DIR_PARAM = None,
    mlflow: MLFLOW_FLAG = False,
    run_name: RUN_NAME_PARAM = None,
) -> None:
    """Evaluate a trained checkpoint: MAE/RMSE/R2 plus regression plots.

    Arguments:
    - `config_path`: Path to TOML configuration file (must define `data.targets`).
    - `checkpoint`: Path to model checkpoint.
    - Override: `--split`, `--batch-size`, `--output-dir`, `--mlflow`, `--run-name`.
    """
    console.print(f"📖 Loading configuration from: {config_path}")
    job = cast(InferenceJobConfig, load_config(config_path, run_type="predict"))

    console.print(f"📊 Evaluating checkpoint: {checkpoint}")
    effective_batch_size = batch_size if batch_size is not None else 32
    result = evaluate_api(
        job,
        checkpoint_path=checkpoint,
        split=cast(Literal["test", "predict"], split),
        log_to_mlflow=mlflow,
        run_name=run_name,
        batch_size=effective_batch_size,
    )

    console.print("🎉 Evaluation completed successfully!")
    present_evaluation_result(result, console, output_dir=output_dir)


@app.command(name="")
def entry(
    config_path: CONFIG_PATH_ARG,
    checkpoint: CHECKPOINT_ARG,
    split: SPLIT_PARAM = "test",
    batch_size: BATCH_SIZE_PARAM = None,
    output_dir: OUTPUT_DIR_PARAM = None,
    mlflow: MLFLOW_FLAG = False,
    run_name: RUN_NAME_PARAM = None,
) -> None:
    """Evaluate a trained checkpoint against labeled data.

    Usage:
    - `dlkit evaluate CONFIG.toml CHECKPOINT.ckpt`
    - `dlkit evaluate CONFIG.toml CHECKPOINT.ckpt --split predict --output-dir out/`
    """
    _run_evaluate_impl(
        config_path=config_path,
        checkpoint=checkpoint,
        split=split,
        batch_size=batch_size,
        output_dir=output_dir,
        mlflow=mlflow,
        run_name=run_name,
    )
