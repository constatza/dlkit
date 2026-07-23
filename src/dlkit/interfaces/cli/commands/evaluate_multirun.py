"""Multirun evaluation command for DLKit CLI.

Batch-evaluates every child run of a multirun/sweep parent MLflow run: fans a
single `evaluate()` call out over each child, matching on the
`mlflow.parentRunId` tag convention. Distinct from `dlkit evaluate`, which
evaluates exactly one checkpoint.
"""

from __future__ import annotations

from typing import Annotated, Literal, cast

import typer
from rich.console import Console

from dlkit.interfaces.inference import evaluate_multirun as evaluate_multirun_api

from ..adapters.config_adapter import load_config
from ..adapters.result_presenter import present_multirun_evaluation_result
from ..middleware.error_handler import handle_cli_errors
from ..params import CONFIG_PATH_ARG, MLFLOW_FLAG, OUTPUT_DIR_PARAM, SPLIT_PARAM

app = typer.Typer(
    name="evaluate-multirun",
    help="📊 Multirun evaluation — Batch-evaluate every child run of a sweep.",
)

console = Console()

PARENT_RUN_ID_PARAM = Annotated[
    str,
    typer.Option(
        "--parent-run-id", help="MLflow run id of the multirun/sweep parent run to evaluate."
    ),
]


@handle_cli_errors(console)
def _run_evaluate_multirun_impl(
    config_path: CONFIG_PATH_ARG,
    parent_run_id: PARENT_RUN_ID_PARAM,
    split: SPLIT_PARAM = "test",
    output_dir: OUTPUT_DIR_PARAM = None,
    mlflow: MLFLOW_FLAG = False,
) -> None:
    """Evaluate every child run of a multirun/sweep parent: MAE/RMSE/R2 plus plots.

    Arguments:
    - `config_path`: Path to TOML configuration file (must define `data.targets`).
    - `parent_run_id`: MLflow run id of the multirun/sweep parent.
    - Override: `--split`, `--output-dir`, `--mlflow`.
    """
    console.print(f"📖 Loading configuration from: {config_path}")
    job = load_config(config_path, run_type="predict")

    console.print(f"📊 Evaluating children of parent run: {parent_run_id}")
    result = evaluate_multirun_api(
        job,
        parent_run_id=parent_run_id,
        split=cast(Literal["test", "predict"], split),
        log_to_mlflow=mlflow,
    )

    console.print("🎉 Multirun evaluation completed successfully!")
    present_multirun_evaluation_result(result, console, output_dir=output_dir)


@app.command(name="")
def entry(
    config_path: CONFIG_PATH_ARG,
    parent_run_id: PARENT_RUN_ID_PARAM,
    split: SPLIT_PARAM = "test",
    output_dir: OUTPUT_DIR_PARAM = None,
    mlflow: MLFLOW_FLAG = False,
) -> None:
    """Batch-evaluate every child run of a multirun/sweep parent run.

    Usage:
    - `dlkit evaluate-multirun CONFIG.toml --parent-run-id abc123`
    - `dlkit evaluate-multirun CONFIG.toml --parent-run-id abc123 --split predict --output-dir out/`
    """
    _run_evaluate_multirun_impl(
        config_path=config_path,
        parent_run_id=parent_run_id,
        split=split,
        output_dir=output_dir,
        mlflow=mlflow,
    )
