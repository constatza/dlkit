"""Shared preparation context for runtime workflow entrypoints."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from dlkit.shared.errors import WorkflowError
from dlkit.tools.io.path_context import get_current_path_context, path_override_context

from ._overrides import apply_runtime_overrides, build_runtime_overrides, validate_runtime_overrides
from ._settings import WorkflowSettings, coerce_general_settings


@dataclass(frozen=True, slots=True)
class EntrypointContext:
    """Prepared runtime settings plus shared execution metadata."""

    settings: Any
    start_time: float = field(default_factory=time.time)
    path_overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def prepare(
        cls,
        raw_settings: WorkflowSettings,
        overrides: Mapping[str, Any] | None,
        *,
        workflow_name: str,
    ) -> EntrypointContext:
        """Coerce settings, validate overrides, and derive path context state."""
        effective = coerce_general_settings(raw_settings)
        normalized_overrides = build_runtime_overrides(**dict(overrides or {}))
        errors = validate_runtime_overrides(**normalized_overrides)
        if errors:
            raise WorkflowError(
                f"Override validation failed: {'; '.join(errors)}",
                {"workflow": workflow_name, "validation_errors": "; ".join(errors)},
            )
        if normalized_overrides:
            effective = apply_runtime_overrides(effective, **normalized_overrides)

        path_overrides: dict[str, Any] = {}
        current_context = get_current_path_context()
        root_from_cfg = getattr(getattr(effective, "SESSION", None), "root_dir", None)
        if root_from_cfg and not (current_context and getattr(current_context, "root_dir", None)):
            path_overrides["root_dir"] = root_from_cfg
        return cls(settings=effective, path_overrides=path_overrides)

    def elapsed(self) -> float:
        """Return elapsed time since preparation."""
        return time.time() - self.start_time

    def run_with_path_context(self, fn):
        """Execute a callback within the derived path override context."""
        if self.path_overrides:
            with path_override_context(self.path_overrides):
                return fn()
        return fn()
