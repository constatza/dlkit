"""Configuration command group for DLKit CLI.

`dlkit config` is a command group with focused subcommands instead of a single
action. This mirrors common CLI practice for config management:
- `dlkit config validate CONFIG.toml` — validate a configuration for a strategy
- `dlkit config show CONFIG.toml` — pretty-print configuration
- `dlkit config create --output config.toml --type training` — scaffold a template
- `dlkit config sync-templates` — keep example/templates in sync

Rationale: Validation, inspection, and scaffolding are distinct operations with
different options, so they are split into subcommands for clarity and ergonomics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer
import yaml
from rich.console import Console
from rich.syntax import Syntax

from dlkit.interfaces.api import generate_template, validate_config
from dlkit.interfaces.cli.templates import TemplateKind

from ..adapters.config_adapter import load_config
from ..middleware.error_handler import handle_api_error
from ._config_display_helpers import as_config_dict, display_config_table
from ._config_fs_helpers import find_project_root

# Create config command group
app = typer.Typer(
    name="config",
    help=(
        "⚙️ Configuration utilities — validate, inspect, and scaffold configs via"
        " dedicated subcommands (validate/show/create/sync-templates)."
    ),
    no_args_is_help=True,
)

console = Console()


@app.command("validate")
def validate_configuration(
    config_path: Annotated[Path, typer.Argument(help="Path to configuration file")],
    strategy: Annotated[
        str | None,
        typer.Option(
            "--strategy",
            "-s",
            help="Strategy to validate for (training, mlflow, optuna, inference)",
        ),
    ] = None,
) -> None:
    """Validate configuration file.

    Examples:
        dlkit config validate config.toml
        dlkit config validate config.toml --strategy mlflow
    """
    try:
        # Load configuration
        console.print(f"📖 Loading configuration from: {config_path}")
        settings = load_config(config_path)
        console.print("✅ Configuration loaded successfully")

        # Auto-detect strategy if not provided
        if strategy is None:
            run_type = getattr(getattr(settings, "run", None), "type", None)
            match run_type:
                case "predict":
                    strategy = "inference"
                case "search":
                    strategy = "optimize"
                case _:
                    strategy = "training"

        console.print(f"🎯 Validating for strategy: [bold]{strategy}[/bold]")

        # Validate configuration
        validate_config(settings)

        # Strategy-specific completeness checks
        if strategy == "inference":
            checkpoint = getattr(getattr(settings, "model", None), "checkpoint", None)
            if not checkpoint:
                console.print("[red]Inference config requires model.checkpoint[/red]")
                raise typer.Exit(1)

        console.print("✅ Configuration is valid!")

    except typer.Exit:
        raise
    except Exception as e:
        # Handle DLKit errors (validation failures, etc.)
        from dlkit.common.errors import DLKitError

        if isinstance(e, DLKitError):
            handle_api_error(e, console)
        else:
            console.print(f"[red]Unexpected error during validation: {e}[/red]")
        raise typer.Exit(1)


@app.command("show")
def show_configuration(
    config_path: Annotated[Path, typer.Argument(help="Path to configuration file")],
    section: Annotated[
        str | None,
        typer.Option("--section", "-s", help="Show specific section (run, model, training, etc.)"),
    ] = None,
    format: Annotated[
        str, typer.Option("--format", "-f", help="Output format (json, yaml, table)")
    ] = "table",
) -> None:
    """Display configuration in a readable format.

    Examples:
        dlkit config show config.toml
        dlkit config show config.toml --section run
        dlkit config show config.toml --format json
    """
    try:
        # Load configuration
        settings = load_config(config_path)
        config_dict = as_config_dict(settings)

        # Filter by section if requested (case-insensitive: supports both "run" and "RUN")
        if section:
            section_key = next(
                (k for k in config_dict if k.lower() == section.lower()),
                None,
            )
            if section_key is not None:
                config_dict = {section_key: config_dict[section_key]}
            else:
                console.print(f"[red]Section '{section}' not found in configuration[/red]")
                available_sections = list(config_dict.keys())
                console.print(f"Available sections: {', '.join(available_sections)}")
                raise typer.Exit(1)

        # Display based on format
        if format == "json":
            import json

            json_str = json.dumps(config_dict, indent=2, default=str)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
            console.print(syntax)

        elif format == "yaml":
            yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
            syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
            console.print(syntax)

        elif format == "table":
            display_config_table(config_dict, console)

        else:
            console.print(f"[red]Unknown format: {format}[/red]")
            console.print("Available formats: json, yaml, table")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error displaying configuration: {e}[/red]")
        raise typer.Exit(1)


@app.command("create")
def create_template(
    output_path: Annotated[Path, typer.Option("--output", "-o", help="Output path for template")],
    template_type: Annotated[
        str,
        typer.Option("--type", "-t", help="Template type (training, inference, mlflow, optuna)"),
    ] = "training",
) -> None:
    """Create a configuration template.

    Examples:
        dlkit config create --output config.toml --type training
        dlkit config create --output mlflow_config.toml --type mlflow
    """
    try:
        # Use new API service instead of static templates
        valid_types = ["training", "inference", "mlflow", "optuna"]

        if template_type not in valid_types:
            console.print(f"[red]Unknown template type: {template_type}[/red]")
            console.print(f"Available types: {', '.join(valid_types)}")
            raise typer.Exit(1)

        # Generate template using API
        template_content = generate_template(cast(TemplateKind, template_type))

        # Write template to file
        with open(output_path, "w") as f:
            f.write(template_content)

        typer.echo(f"✅ Template created: {output_path}")
        typer.echo(f"📝 Template type: {template_type}")
        typer.echo("\n💡 Next steps:")
        typer.echo(f"  1. Edit {output_path} with your specific settings")
        typer.echo(f"  2. Validate: dlkit config validate {output_path}")
        typer.echo(f"  3. Run: dlkit train {output_path}")

    except Exception as e:
        console.print(f"[red]Error creating template: {e}[/red]")
        raise typer.Exit(1)


@app.command("sync-templates")
def sync_templates(
    root: Annotated[Path | None, typer.Option("--root", help="Project root directory")] = None,
    check: Annotated[
        bool, typer.Option("--check", help="Check for drift; non-zero exit if different")
    ] = False,
    write: Annotated[bool, typer.Option("--write", help="Write updated templates to disk")] = False,
) -> None:
    """Synchronize example and template TOML files with canonical builders.

    Without options, performs a check; use --write to update files.
    """
    try:
        root_dir = find_project_root(root)

        # Use API to generate templates instead of static ones
        targets: list[tuple[str, Path]] = [
            (generate_template("training"), root_dir / "example_config.toml"),
            (generate_template("training"), root_dir / "config" / "templates" / "training.toml"),
            (generate_template("inference"), root_dir / "config" / "templates" / "inference.toml"),
            (generate_template("mlflow"), root_dir / "config" / "templates" / "mlflow.toml"),
            (generate_template("optuna"), root_dir / "config" / "templates" / "optuna.toml"),
        ]

        has_drift = False
        for content, path in targets:
            existing = path.read_text() if path.exists() else None
            if existing != content:
                has_drift = True
                if write:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content)
                    console.print(f"✍️  Updated {path}")

        if check and has_drift and not write:
            console.print("[red]Template drift detected[/red]")
            raise typer.Exit(1)

        if write and not has_drift:
            console.print("✅ Templates already up to date")
    except Exception as e:
        console.print(f"[red]Error syncing templates: {e}[/red]")
        raise typer.Exit(1)
