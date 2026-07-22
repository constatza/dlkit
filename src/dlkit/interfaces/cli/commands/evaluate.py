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

from dlkit.common import ConfigurationError
from dlkit.common.checkpoint_source import CheckpointSource, LatestRunCheckpoint, RunCheckpoint
from dlkit.infrastructure.config.job_config import InferenceJobConfig
from dlkit.interfaces.inference import evaluate as evaluate_api

from ..adapters.config_adapter import load_config
from ..adapters.result_presenter import present_evaluation_result
from ..middleware.error_handler import handle_cli_errors
from ..params import (
    BATCH_SIZE_PARAM,
    CONFIG_PATH_ARG,
    MLFLOW_FLAG,
    OUTPUT_DIR_PARAM,
    RUN_NAME_PARAM,
    SPLIT_PARAM,
)

app = typer.Typer(
    name="evaluate",
    help="📊 Evaluation — Stats/plots for a trained checkpoint against labeled data.",
)

console = Console()

# Local optional checkpoint argument for `evaluate` specifically: unlike
# `predict`, a checkpoint here may instead be resolved from an MLflow run via
# `--run-id`/`--latest-run`, so it can't be the shared, required
# `params.CHECKPOINT_ARG` (still required for `predict.py`).
CHECKPOINT_OPT_ARG = Annotated[
    Path | None,
    typer.Argument(
        help="Path to model checkpoint. Omit when using --run-id or --latest-run instead."
    ),
]

RUN_ID_PARAM = Annotated[
    str | None,
    typer.Option("--run-id", help="Resolve the checkpoint from this exact MLflow run id."),
]

LATEST_RUN_FLAG = Annotated[
    bool,
    typer.Option(
        "--latest-run",
        help="Resolve the checkpoint from the most recently started MLflow run "
        "in the configured experiment.",
    ),
]


def _resolve_checkpoint_selection(
    checkpoint: Path | None,
    run_id: str | None,
    latest_run: bool,
) -> tuple[Path | None, CheckpointSource | None]:
    """Translate CLI checkpoint-selection flags into exactly one selector.

    Args:
        checkpoint: Positional checkpoint path, if given.
        run_id: `--run-id` value, if given.
        latest_run: `--latest-run` flag value.

    Returns:
        A `(checkpoint_path, run_checkpoint)` pair suitable for `evaluate()`,
        with at most one side set.

    Raises:
        ConfigurationError: More than one of `checkpoint`, `run_id`,
            `latest_run` was given — these are mutually exclusive
            checkpoint-selection methods.
    """
    selected = sum((checkpoint is not None, run_id is not None, latest_run))
    if selected > 1:
        raise ConfigurationError(
            "Pass at most one checkpoint-selection method: CHECKPOINT, --run-id, or --latest-run.",
            {
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "run_id": run_id,
                "latest_run": latest_run,
            },
        )
    if run_id is not None:
        return None, RunCheckpoint(run_id=run_id)
    if latest_run:
        return None, LatestRunCheckpoint()
    return checkpoint, None


@handle_cli_errors(console)
def _run_evaluate_impl(
    config_path: CONFIG_PATH_ARG,
    checkpoint: CHECKPOINT_OPT_ARG = None,
    run_id: RUN_ID_PARAM = None,
    latest_run: LATEST_RUN_FLAG = False,
    split: SPLIT_PARAM = "test",
    batch_size: BATCH_SIZE_PARAM = None,
    output_dir: OUTPUT_DIR_PARAM = None,
    mlflow: MLFLOW_FLAG = False,
    run_name: RUN_NAME_PARAM = None,
) -> None:
    """Evaluate a trained checkpoint: MAE/RMSE/R2 plus regression plots.

    Arguments:
    - `config_path`: Path to TOML configuration file (must define `data.targets`).
    - `checkpoint`: Path to model checkpoint. Mutually exclusive with `--run-id`
      and `--latest-run`.
    - Override: `--run-id`, `--latest-run`, `--split`, `--batch-size`,
      `--output-dir`, `--mlflow`, `--run-name`.
    """
    console.print(f"📖 Loading configuration from: {config_path}")
    job = cast(InferenceJobConfig, load_config(config_path, run_type="predict"))

    checkpoint_path, run_checkpoint = _resolve_checkpoint_selection(checkpoint, run_id, latest_run)

    match run_checkpoint:
        case RunCheckpoint(run_id=selected_run_id):
            console.print(f"📊 Evaluating checkpoint from run: {selected_run_id}")
        case LatestRunCheckpoint():
            console.print("📊 Evaluating checkpoint from latest run")
        case None:
            console.print(f"📊 Evaluating checkpoint: {checkpoint_path}")

    effective_batch_size = batch_size if batch_size is not None else 32
    result = evaluate_api(
        job,
        checkpoint_path=checkpoint_path,
        run_checkpoint=run_checkpoint,
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
    checkpoint: CHECKPOINT_OPT_ARG = None,
    run_id: RUN_ID_PARAM = None,
    latest_run: LATEST_RUN_FLAG = False,
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
    - `dlkit evaluate CONFIG.toml --run-id abc123`
    - `dlkit evaluate CONFIG.toml --latest-run`
    """
    _run_evaluate_impl(
        config_path=config_path,
        checkpoint=checkpoint,
        run_id=run_id,
        latest_run=latest_run,
        split=split,
        batch_size=batch_size,
        output_dir=output_dir,
        mlflow=mlflow,
        run_name=run_name,
    )
