"""Strict runtime override payload models for workflow entrypoints."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class RuntimeOverrideModel(BaseModel):
    """Base class for request-scoped workflow overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_runtime_kwargs(self) -> dict[str, object]:
        """Return only explicitly provided override values."""
        return self.model_dump(exclude_none=True)


def require_override_model[T: RuntimeOverrideModel](
    overrides: object,
    model_type: type[T],
) -> T | None:
    """Require a strict override model instance when overrides are provided."""
    if overrides is None:
        return None
    if isinstance(overrides, model_type):
        return overrides
    raise TypeError(
        f"overrides must be provided as {model_type.__name__}, got {type(overrides).__name__}"
    )


class TrainingOverrides(RuntimeOverrideModel):
    """Supported runtime overrides for training entrypoints."""

    checkpoint_path: Path | None = None
    root_dir: Path | None = None
    output_dir: Path | None = None
    data_dir: Path | None = None
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    experiment_name: str | None = None
    run_name: str | None = None
    register_model: bool | None = None
    tags: dict[str, str] | None = None
    loss_function: str | None = None
    loss_module: str | None = None


class OptimizationOverrides(RuntimeOverrideModel):
    """Supported runtime overrides for optimization entrypoints."""

    checkpoint_path: Path | None = None
    root_dir: Path | None = None
    output_dir: Path | None = None
    data_dir: Path | None = None
    trials: int | None = None
    study_name: str | None = None
    experiment_name: str | None = None
    run_name: str | None = None
    enable_optuna: bool | None = None
    register_model: bool | None = None
    tags: dict[str, str] | None = None


class ExecutionOverrides(TrainingOverrides):
    """Superset of supported overrides for the unified execution entrypoint."""

    trials: int | None = None
    study_name: str | None = None
    enable_optuna: bool | None = None

    def to_training_overrides(self) -> TrainingOverrides:
        """Project the unified payload onto training-only fields."""
        keys = set(TrainingOverrides.model_fields)
        return TrainingOverrides.model_validate(
            {key: value for key, value in self.to_runtime_kwargs().items() if key in keys}
        )

    def to_optimization_overrides(self) -> OptimizationOverrides:
        """Project the unified payload onto optimization-only fields."""
        keys = set(OptimizationOverrides.model_fields)
        return OptimizationOverrides.model_validate(
            {key: value for key, value in self.to_runtime_kwargs().items() if key in keys}
        )
