"""Core API functions for training, optimization, and inference."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from dlkit.interfaces.api.domain.override_types import (
    OptimizationOverrides,
    TrainingOverrides,
)
from dlkit.runtime.workflows.entrypoints import optimize as runtime_optimize
from dlkit.runtime.workflows.entrypoints import train as runtime_train
from dlkit.runtime.workflows.entrypoints._settings import WorkflowSettings
from dlkit.shared import (
    OptimizationResult,
    TrainingResult,
)


def _coerce_override_paths(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Coerce known path-like override values into ``Path`` objects."""
    path_fields = {"checkpoint_path", "root_dir", "output_dir", "data_dir"}
    coerced: dict[str, Any] = {}
    if overrides is not None:
        for key, value in overrides.items():
            coerced[key] = value
    for field in path_fields:
        value = coerced.get(field)
        if isinstance(value, str):
            coerced[field] = Path(value)
    return coerced


def train(
    settings: WorkflowSettings,
    overrides: TrainingOverrides | None = None,
) -> TrainingResult:
    """Run training with optional overrides."""
    return runtime_train(
        settings,
        overrides=cast(Any, _coerce_override_paths(overrides)),
    )


# REMOVED: Old infer() and predict_with_config() functions
# Use the new load_model() API instead:
#
#   from dlkit import load_model
#   predictor = load_model("model.ckpt", device="cuda")
#   result = predictor.predict(inputs)
#
# See documentation for migration guide.


def optimize(
    settings: WorkflowSettings,
    overrides: OptimizationOverrides | None = None,
) -> OptimizationResult:
    """Run Optuna hyperparameter optimization with optional overrides."""
    return runtime_optimize(
        settings,
        overrides=cast(Any, _coerce_override_paths(overrides)),
    )
