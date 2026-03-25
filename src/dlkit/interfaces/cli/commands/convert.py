"""CLI command to convert/export checkpoints to ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from dlkit.interfaces.api.commands.convert_command import ConvertCommand, ConvertCommandInput
from dlkit.interfaces.cli.adapters.config_adapter import load_config

app = typer.Typer(name="convert", help="🔁 Convert checkpoints to export formats (e.g., ONNX)")
console = Console()


@app.command(name="")
def entry(
    checkpoint: Annotated[Path, typer.Argument(help="Path to model checkpoint (.ckpt)")],
    output: Annotated[Path, typer.Argument(help="Output ONNX file path (.onnx)")],
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Optional config to infer input shape")
    ] = None,
    shape: Annotated[
        str | None,
        typer.Option(
            "--shape",
            "-s",
            help="Feature dims only (no batch), e.g. '3,224,224' or multiple via '3,224,224;10'",
        ),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            "-b",
            help="Batch size to prefix for --shape, or to validate against dataloader when using --config",
            show_default=True,
        ),
    ] = 1,
    opset: Annotated[
        int, typer.Option("--opset", help="ONNX opset version", show_default=True)
    ] = 17,
) -> None:
    """Export a Lightning checkpoint to an ONNX model.

    Examples:
      dlkit convert model.ckpt model.onnx --shape 3,224,224
      dlkit convert model.ckpt model.onnx -c config.toml
      dlkit convert model.ckpt model.onnx --shape 3,32,32 --opset 17 --batch-size 4
    """
    try:
        # Require either --shape or --config
        if not shape and not config:
            raise typer.BadParameter(
                "Provide either --shape (full dims incl. batch) or --config for dataloader-based inference"
            )

        # Load settings only if provided (for dataloader-based inference)
        settings = load_config(config) if config else None

        if settings is None and not shape:
            raise typer.BadParameter("Provide either --shape or --config")

        cmd = ConvertCommand()
        result = cmd.execute(
            ConvertCommandInput(
                checkpoint_path=checkpoint,
                output_path=output,
                shape=shape,
                opset=opset,
                batch_size=batch_size,
            ),
            cast(Any, settings),
        )

        text = Text()
        text.append("✅ Export successful\n\n", style="bold green")
        # Use POSIX formatting so Rich output stays consistent across platforms
        formatted_output = Path(result.output_path).as_posix()
        text.append(f"Output: {formatted_output}\n", style="cyan")
        text.append(f"Opset: {result.opset}\n")
        text.append(f"Inputs: {', '.join(str(s) for s in result.inputs)}\n")
        panel = Panel.fit(text, title="ONNX Export", border_style="green")
        console.print(panel)

    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")
        raise typer.Exit(1)
