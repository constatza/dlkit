"""Public override payload types for the API layer."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict


class TrainingOverrides(TypedDict, total=False):
    """Supported API overrides for training workflows."""

    checkpoint_path: Path | str
    root_dir: Path | str
    output_dir: Path | str
    data_dir: Path | str
    epochs: int
    batch_size: int
    learning_rate: float
    experiment_name: str
    run_name: str
    register_model: bool
    tags: dict[str, str]
    loss_function: str
    loss_module: str


class OptimizationOverrides(TypedDict, total=False):
    """Supported API overrides for optimization workflows."""

    checkpoint_path: Path | str
    root_dir: Path | str
    output_dir: Path | str
    data_dir: Path | str
    trials: int
    study_name: str
    experiment_name: str
    run_name: str
    enable_optuna: bool
    register_model: bool
    tags: dict[str, str]


class ExecutionOverrides(TrainingOverrides, total=False):
    """Superset of supported API overrides for the unified execution entrypoint."""

    trials: int
    study_name: str


__all__ = [
    "ExecutionOverrides",
    "OptimizationOverrides",
    "TrainingOverrides",
]
