"""Prediction command for DLKit CLI.

Usage patterns:
- Provide checkpoint via CLI: `dlkit predict CONFIG.toml CHECKPOINT [options]`
- Or rely on config: set `[MODEL].checkpoint` in `CONFIG.toml`, then pass
  `CHECKPOINT` as the same value or omit the CLI argument in future versions.

Notes:
- This uses stateful predictor-based inference with training configurations and datasets
- For direct inference without config files, use the Python API: dlkit.load_model()
- Today the CLI requires a `CHECKPOINT` argument; if your config contains
  `[MODEL].checkpoint`, the CLI argument will take precedence when provided.
  This keeps behavior explicit while remaining compatible with common practice.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from dlkit.interfaces.inference.api import load_model
from dlkit.runtime.workflows.factories.inference_data_factory import build_inference_datamodule
from dlkit.tools.config.workflow_configs import InferenceWorkflowConfig

from ..adapters.config_adapter import load_config
from ..adapters.result_presenter import present_inference_result
from ..middleware.error_handler import handle_api_error
from ..params import (
    BATCH_SIZE_PARAM,
    CHECKPOINT_ARG,
    CONFIG_PATH_ARG,
    DATA_DIR_PARAM,
    OUTPUT_DIR_PARAM,
    ROOT_DIR_PARAM,
    SAVE_PREDICTIONS_FLAG,
)

# Single prediction command group (no subcommands)
app = typer.Typer(
    name="predict",
    help=(
        "🔮 Prediction — Run Lightning-based predictions with training configs. "
        "Checkpoint can be supplied via CLI or config ([MODEL].checkpoint)."
    ),
)

console = Console()


def _run_inference_impl(
    config_path: CONFIG_PATH_ARG,
    checkpoint: CHECKPOINT_ARG,
    root_dir: ROOT_DIR_PARAM = None,
    output_dir: OUTPUT_DIR_PARAM = None,
    data_dir: DATA_DIR_PARAM = None,
    batch_size: BATCH_SIZE_PARAM = None,
    save_predictions: SAVE_PREDICTIONS_FLAG = True,
) -> None:
    """Run inference using a trained model with parameter overrides.

    Arguments:
    - `config_path`: Path to TOML configuration file.
    - `checkpoint`: Path to model checkpoint. If the configuration contains
      `[MODEL].checkpoint`, that value is used by the API when this argument
      is not provided (future behavior). When both are present, this argument
      takes precedence.
    - Overrides: `--output-dir`, `--dataflow-dir`, `--batch-size`.
    """
    try:
        # Load configuration first to resolve optional checkpoint
        console.print(f"📖 Loading configuration from: {config_path}")
        settings = load_config(config_path, root_dir=root_dir, workflow_type="inference")

        # Resolve checkpoint: CLI argument wins; otherwise, use config [MODEL].checkpoint
        effective_checkpoint: Path | None = checkpoint
        try:
            if effective_checkpoint is None and getattr(settings, "MODEL", None) is not None:
                cfg_ckpt = getattr(settings.MODEL, "checkpoint", None)
                if cfg_ckpt:
                    from pathlib import Path as _P

                    effective_checkpoint = _P(str(cfg_ckpt))
        except Exception:
            pass

        if effective_checkpoint is None or not effective_checkpoint.exists():
            console.print(f"[red]Checkpoint file not found: {checkpoint}[/red]")
            raise typer.Exit(1)

        # Show applied overrides
        override_messages = []
        if output_dir:
            override_messages.append(f"Output dir: {output_dir}")
        if data_dir:
            override_messages.append(f"Data dir: {data_dir}")
        if batch_size:
            override_messages.append(f"Batch size: {batch_size}")

        if root_dir:
            override_messages.append(f"Root dir: {root_dir}")
        if override_messages:
            console.print("🔧 Parameter overrides:")
            for msg in override_messages:
                console.print(f"  • {msg}")

        console.print(f"🔮 Loading model from checkpoint: {effective_checkpoint}")

        # Execute inference using new stateful predictor API
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Load predictor once
            load_task = progress.add_task("Loading predictor...", total=None)

            effective_batch_size = batch_size if batch_size is not None else 32

            predictor = load_model(
                checkpoint_path=effective_checkpoint,
                device="auto",
                batch_size=effective_batch_size,
                apply_transforms=True,
                auto_load=True,
            )
            progress.remove_task(load_task)

            # Build datamodule from settings, iterate batches, call predict()
            inference_task = progress.add_task("Running inference...", total=None)

            # Get ordered feature names for kwarg dispatch (from checkpoint metadata)
            feature_names: tuple[str, ...] = (
                predictor._model_state.feature_names if predictor._model_state is not None else ()
            )

            datamodule = (
                build_inference_datamodule(cast(InferenceWorkflowConfig, settings))
                if getattr(settings, "has_batch_inference_config", False)
                else None
            )
            if datamodule is not None:
                datamodule.setup("predict")
                loader = datamodule.predict_dataloader()
            else:
                loader = []

            import torch as _torch

            all_predictions = []
            for batch in loader:
                features_td = batch["features"]
                if feature_names:
                    feature_kwargs = {
                        name: features_td[name]
                        for name in feature_names
                        if name in features_td.keys()
                    }
                else:
                    # Fallback: use all feature keys from batch
                    feature_kwargs = {k: features_td[k] for k in features_td.keys()}

                output = predictor.predict(**feature_kwargs)
                # Extract primary prediction (first element of tuple, or the tensor itself)
                prediction = output[0] if isinstance(output, tuple) else output
                if isinstance(prediction, _torch.Tensor):
                    all_predictions.append(prediction)

            progress.remove_task(inference_task)

            # Unload predictor to free resources
            predictor.unload()

        # Combine predictions from all batches (all elements are Tensors)
        import torch

        predictions = torch.cat(all_predictions, dim=0) if all_predictions else None

        # Create InferenceResult for presentation
        from dlkit.interfaces.api.domain.models import InferenceResult

        result = InferenceResult(
            model_state=None, predictions=predictions, metrics=None, duration_seconds=0.0
        )

        console.print("🎉 Inference completed successfully!")

        # Present results
        present_inference_result(result, console, save_predictions=save_predictions)

    except typer.Exit:
        raise
    except Exception as e:
        # Handle DLKit errors (inference failures, etc.)
        from dlkit.interfaces.api.domain.errors import DLKitError

        if isinstance(e, DLKitError):
            handle_api_error(e, console)
        else:
            console.print(f"[red]Unexpected error during inference: {e}[/red]")
        raise typer.Exit(1)


# Default inference entry: dlkit infer CONFIG.toml CHECKPOINT [options]
@app.command(name="")
def entry(
    config_path: CONFIG_PATH_ARG,
    checkpoint: CHECKPOINT_ARG,
    root_dir: ROOT_DIR_PARAM = None,
    output_dir: OUTPUT_DIR_PARAM = None,
    data_dir: DATA_DIR_PARAM = None,
    batch_size: BATCH_SIZE_PARAM = None,
    save_predictions: SAVE_PREDICTIONS_FLAG = True,
) -> None:
    """Run inference with configuration and parameter overrides.

    Usage examples:
    - `dlkit infer CONFIG.toml CHECKPOINT.ckpt`
    - `dlkit infer CONFIG.toml --output-dir ./out --batch-size 64` (uses [MODEL].checkpoint)

    Note: Supplying `CHECKPOINT` via CLI is explicit and recommended. If
    `[MODEL].checkpoint` is present in the configuration, it may be used by
    the underlying API; when both are specified, the CLI argument wins.
    """
    _run_inference_impl(
        config_path=config_path,
        checkpoint=checkpoint,
        root_dir=root_dir,
        output_dir=output_dir,
        data_dir=data_dir,
        batch_size=batch_size,
        save_predictions=save_predictions,
    )
