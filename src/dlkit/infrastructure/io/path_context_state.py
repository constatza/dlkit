"""Neutral path-context state shared by config and I/O helpers.

Uses threading.local() for thread-local state, guaranteeing sync-only semantics.
For async code, migrate to contextvars.ContextVar instead.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True, kw_only=True)
class PathOverrideContext:
    """Thread-local path override state."""

    root_dir: Path | None = None
    output_dir: Path | None = None
    data_dir: Path | None = None
    checkpoints_dir: Path | None = None


_context_storage = threading.local()


def get_current_path_context() -> PathOverrideContext | None:
    """Return the active path override context for the current thread."""

    return getattr(_context_storage, "path_context", None)


def set_path_context(context: PathOverrideContext | None) -> None:
    """Set or clear the active path override context for the current thread."""

    _context_storage.path_context = context
