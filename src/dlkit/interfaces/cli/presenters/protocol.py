"""Shared protocol for CLI result presenters."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from rich.console import Console


@runtime_checkable
class IResultPresenter(Protocol):
    """Protocol for presenting results to a Rich console."""

    def present(self, result: Any, console: Console) -> None:
        """Render a result to the provided console."""
