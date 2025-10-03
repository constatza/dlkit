"""Prediction command for DLKit CLI.

Usage patterns:
- Provide checkpoint via CLI: `dlkit predict CONFIG.toml CHECKPOINT [options]`
- Or rely on config: set `[MODEL].checkpoint` in `CONFIG.toml`, then pass
  `CHECKPOINT` as the same value or omit the CLI argument in future versions.

Notes:
- This uses Lightning-based prediction with training configurations and datasets
- For direct inference without config files, use the Python API: dlkit.interfaces.api.infer()
- Today the CLI requires a `CHECKPOINT` argument; if your config contains
  `[MODEL].checkpoint`, the CLI argument will take precedence when provided.
  This keeps behavior explicit while remaining compatible with common practice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from dlkit.interfaces.api import predict_with_config as api_predict_with_config

from ..adapters.config_adapter import load_config
from ..adapters.result_presenter import present_inference_result
from ..middleware.error_handler import handle_api_error

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
    checkpoint: Annotated[Path, typer.Argument(help="Path to model checkpoint")],
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
    batch_size: Annotated[
        int | None, typer.Option("--batch-size", "-b", help="Override batch size for inference")
    ] = None,
    save_predictions: Annotated[
        bool, typer.Option("--save", "-s", help="Save predictions to file")
    ] = True,
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
        settings = load_config(config_path, root_dir=root_dir, workflow_type="training")

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

        # Execute inference with overrides
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running inference...", total=None)
            inference_result = api_predict_with_config(
                training_settings=settings,
                checkpoint_path=effective_checkpoint,
                root_dir=root_dir,
                output_dir=output_dir,
                data_dir=data_dir,
                batch_size=batch_size,
            )
            progress.remove_task(task)

        result = inference_result
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
    checkpoint: Annotated[Path, typer.Argument(help="Path to model checkpoint")],
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
    batch_size: Annotated[
        int | None, typer.Option("--batch-size", "-b", help="Override batch size for inference")
    ] = None,
    save_predictions: Annotated[
        bool, typer.Option("--save", "-s", help="Save predictions to file")
    ] = True,
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
