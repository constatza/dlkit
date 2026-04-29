"""Shared runtime override payload models."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class _StrictOverrideModel(BaseModel):
    """Base class for strict runtime override payloads."""

    model_config = ConfigDict(extra="forbid")


class TrainingOverrides(_StrictOverrideModel):
    """Supported overrides for training workflows."""

    checkpoint_path: Path | str | None = None
    root_dir: Path | str | None = None
    output_dir: Path | str | None = None
    data_dir: Path | str | None = None
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    experiment_name: str | None = None
    run_name: str | None = None
    register_model: bool | None = None
    tags: dict[str, str] | None = None
    loss_function: str | None = None
    loss_module: str | None = None


class OptimizationOverrides(_StrictOverrideModel):
    """Supported overrides for optimization workflows."""

    checkpoint_path: Path | str | None = None
    root_dir: Path | str | None = None
    output_dir: Path | str | None = None
    data_dir: Path | str | None = None
    trials: int | None = None
    study_name: str | None = None
    experiment_name: str | None = None
    run_name: str | None = None
    enable_optuna: bool | None = None
    register_model: bool | None = None
    tags: dict[str, str] | None = None


class ExecutionOverrides(TrainingOverrides):
    """Superset of supported overrides for unified execution."""

    trials: int | None = None
    study_name: str | None = None
    enable_optuna: bool | None = None
