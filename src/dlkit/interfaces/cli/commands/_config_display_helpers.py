"""Presentation helpers for `dlkit config` commands."""

from __future__ import annotations

from typing import Any, cast

from rich.console import Console
from rich.table import Table


def as_config_dict(obj: Any) -> dict[str, Any]:
    """Serialize a settings object to a plain dict for display."""
    try:
        fn = getattr(obj, "to_dict", None)
        if callable(fn):
            data = fn()
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    try:
        model_dump = getattr(obj, "model_dump", None)
        if callable(model_dump):
            return cast(dict[str, Any], model_dump(exclude_none=True))
    except Exception:
        pass
    try:
        return dict(obj)
    except Exception:
        return {}


def add_config_rows(table: Table, data: dict[str, Any], prefix: str = "") -> None:
    """Recursively add configuration rows to a Rich table."""
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            table.add_row(f"[bold]{full_key}[/bold]", "[dim]<section>[/dim]", "dict")
            add_config_rows(table, value, full_key)
            continue

        value_str = str(value)
        if len(value_str) > 50:
            value_str = value_str[:47] + "..."
        table.add_row(full_key, value_str, type(value).__name__)


def display_config_table(
    config_dict: dict[str, Any],
    console: Console,
    parent_key: str = "",
) -> None:
    """Display configuration as a hierarchical table."""
    table = Table(title="Configuration" if not parent_key else f"Configuration: {parent_key}")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Type", style="yellow")
    add_config_rows(table, config_dict)
    console.print(table)
