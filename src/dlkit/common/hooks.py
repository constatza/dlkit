"""Shared workflow hook contracts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class LifecycleHooks:
    """Functional extension points for lifecycle events."""

    on_run_created: Callable[[str, str | None], None] | None = field(default=None)
    on_training_complete: Callable[[Any], None] | None = field(default=None)
    extra_tags: Callable[[Any], dict[str, str]] | None = field(default=None)
    extra_params: Callable[[Any], dict[str, Any]] | None = field(default=None)
    extra_artifacts: Callable[[Any], Sequence[Path]] | None = field(default=None)
