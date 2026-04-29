"""Shared preparation context for runtime workflow entrypoints."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from dlkit.common.errors import WorkflowError
from dlkit.infrastructure.config.workflow_types import WorkflowConfig
from dlkit.infrastructure.io.path_context import get_current_path_context, path_override_context

from ._override_types import RuntimeOverrideModel
from ._overrides import apply_runtime_overrides, build_runtime_overrides, validate_runtime_overrides
from ._settings import WorkflowSettings

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class EntrypointContext:
    """Prepared runtime settings plus shared execution metadata."""

    settings: WorkflowConfig
    start_time: float = field(default_factory=time.time)
    path_overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _normalize_overrides(
        cls, overrides: Mapping[str, Any] | RuntimeOverrideModel | None
    ) -> dict[str, Any]:
        """Normalize override payloads to a plain dict for runtime helpers."""
        if overrides is None:
            return {}
        if isinstance(overrides, RuntimeOverrideModel):
            return overrides.to_runtime_kwargs()
        return dict(overrides)

    @classmethod
    def prepare(
        cls,
        raw_settings: WorkflowSettings,
        overrides: Mapping[str, Any] | RuntimeOverrideModel | None,
        *,
        workflow_name: str,
    ) -> EntrypointContext:
        """Validate overrides and derive path context state."""
        effective = raw_settings
        normalized_overrides = build_runtime_overrides(**cls._normalize_overrides(overrides))
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
        already_has_root = current_context and getattr(current_context, "root_dir", None)
        if not already_has_root:
            # Prefer an explicit root_dir override; fall back to SESSION.root_dir
            override_root = normalized_overrides.get("root_dir")
            root_from_cfg = getattr(getattr(effective, "SESSION", None), "root_dir", None)
            resolved_root = override_root or root_from_cfg
            if resolved_root:
                path_overrides["root_dir"] = resolved_root
        return cls(settings=effective, path_overrides=path_overrides)

    def elapsed(self) -> float:
        """Return elapsed time since preparation."""
        return time.time() - self.start_time

    def run_with_path_context(self, fn: Callable[[], T]) -> T:
        """Execute a callback within the derived path override context."""
        if self.path_overrides:
            with path_override_context(self.path_overrides):
                return fn()
        return fn()
