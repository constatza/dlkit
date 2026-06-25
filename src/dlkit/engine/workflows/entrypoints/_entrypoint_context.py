"""Shared preparation context for runtime workflow entrypoints."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar, cast

from dlkit.common.errors import WorkflowError
from dlkit.infrastructure.config.job_config import (
    InferenceJobConfig,
    JobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)

from ._override_types import RuntimeOverrideModel
from ._overrides import apply_runtime_overrides, build_runtime_overrides, validate_runtime_overrides
from ._settings import WorkflowSettings

T = TypeVar("T")

type _AnyOverrideValue = str | int | float | bool | None | dict[str, str]


@dataclass(frozen=True, slots=True)
class EntrypointContext:
    """Prepared runtime settings plus shared execution metadata."""

    settings: TrainingJobConfig | SearchJobConfig | InferenceJobConfig | JobConfig
    start_time: float = field(default_factory=time.time)

    @classmethod
    def _normalize_overrides(
        cls, overrides: Mapping[str, _AnyOverrideValue] | RuntimeOverrideModel | None
    ) -> dict[str, _AnyOverrideValue]:
        """Normalize override payloads to a plain dict for runtime helpers."""
        if overrides is None:
            return {}
        if isinstance(overrides, RuntimeOverrideModel):
            return cast(dict[str, _AnyOverrideValue], overrides.to_runtime_kwargs())
        return dict(overrides)

    @classmethod
    def prepare(
        cls,
        raw_settings: WorkflowSettings,
        overrides: Mapping[str, _AnyOverrideValue] | RuntimeOverrideModel | None,
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

        return cls(settings=effective)

    def elapsed(self) -> float:
        """Return elapsed time since preparation."""
        return time.time() - self.start_time

    def run_with_path_context(self, fn: Callable[[], T]) -> T:
        """Execute a callback within the derived runtime context."""
        return fn()
