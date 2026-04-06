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
from typing import Annotated, Any, cast

import typer
import yaml
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from dlkit.interfaces.api import generate_template, validate_config
from dlkit.interfaces.cli.templates import TemplateKind

from .. import templates as tmpl
from ..adapters.config_adapter import load_config
from ..middleware.error_handler import handle_api_error

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
            if settings.SESSION and settings.SESSION.inference:
                strategy = "inference"
            else:
                strategy = "training"  # Default

        console.print(f"🎯 Validating for strategy: [bold]{strategy}[/bold]")

        # Validate configuration
        validate_config(settings)
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
        typer.Option("--section", "-s", help="Show specific section (SESSION, MODEL, etc.)"),
    ] = None,
    format: Annotated[
        str, typer.Option("--format", "-f", help="Output format (json, yaml, table)")
    ] = "table",
) -> None:
    """Display configuration in a readable format.

    Examples:
        dlkit config show config.toml
        dlkit config show config.toml --section SESSION
        dlkit config show config.toml --format json
    """
    try:
        # Load configuration
        settings = load_config(config_path)

        # Get configuration dict with robust fallback for mocked settings
        def _as_config_dict(obj: Any) -> dict[str, Any]:
            # Preferred: use to_dict() if it returns a dict
            try:
                fn = getattr(obj, "to_dict", None)
                if callable(fn):
                    d = fn()
                    if isinstance(d, dict):
                        return d
            except Exception:
                pass
            # Fallback: Pydantic's model_dump if available
            try:
                md = getattr(obj, "model_dump", None)
                if callable(md):
                    return cast(dict[str, Any], md(exclude_none=True))
            except Exception:
                pass
            # Last resort
            try:
                return dict(obj)
            except Exception:
                return {}

        config_dict = _as_config_dict(settings)

        # Filter by section if requested
        if section:
            if section.upper() in config_dict:
                config_dict = {section.upper(): config_dict[section.upper()]}
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
            _display_config_table(config_dict, console)

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


def _find_project_root(start: Path | None = None) -> Path:
    start = start or Path.cwd()
    cur = start.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return cur


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
        root_dir = _find_project_root(root)

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


def _display_config_table(config_dict: dict, console: Console, parent_key: str = "") -> None:
    """Display configuration as a hierarchical table."""
    table = Table(title="Configuration" if not parent_key else f"Configuration: {parent_key}")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Type", style="yellow")

    def _add_rows(data: dict, prefix: str = ""):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                table.add_row(f"[bold]{full_key}[/bold]", "[dim]<section>[/dim]", "dict")
                _add_rows(value, full_key)
            else:
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                table.add_row(full_key, value_str, type(value).__name__)

    _add_rows(config_dict)
    console.print(table)


def _create_training_template() -> str:
    """Create a basic training configuration template."""
    return tmpl.render_template("training")


def _create_inference_template() -> str:
    """Create an inference configuration template."""
    return tmpl.render_template("inference")


def _create_mlflow_template() -> str:
    """Create an MLflow training configuration template."""
    return tmpl.render_template("mlflow")


def _create_optuna_template() -> str:
    """Create an Optuna optimization configuration template."""
    return tmpl.render_template("optuna")
