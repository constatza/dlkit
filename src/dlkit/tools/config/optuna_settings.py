"""Optuna settings - flattened top-level configuration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import Field, PositiveInt

from .core.base_settings import BasicSettings, ComponentSettings


class PrunerSettings(ComponentSettings):
    """Optuna pruner configuration settings.

    Args:
        component_name: Pruner algorithm name
        module_path: Module path for the pruner
        n_warmup_steps: Number of warmup steps before pruning
        interval_steps: Interval between pruning steps
    """

    name: str | Callable[..., Any] | dict[str, Any] | None = Field(
        default="NopPruner", description="Pruner algorithm name for hyperparameter optimization"
    )
    module_path: str | None = Field(
        default="optuna.pruners", description="Module path for the pruner"
    )
    n_warmup_steps: int | None = Field(
        default=None, description="Number of warmup steps before pruning starts"
    )
    interval_steps: int | None = Field(default=None, description="Interval between pruning steps")


class SamplerSettings(ComponentSettings):
    """Optuna sampler configuration settings.

    Args:
        component_name: Sampler algorithm name
        module_path: Module path for the sampler
        seed: Random seed for reproducibility
    """

    name: str | Callable[..., Any] | dict[str, Any] | None = Field(
        default="TPESampler", description="Sampler algorithm name for hyperparameter optimization"
    )
    module_path: str | None = Field(
        default="optuna.samplers", description="Module path for the sampler"
    )
    seed: int | None = Field(default=None, description="Optional random seed for reproducibility")


class OptunaSettings(BasicSettings):
    """Top-level Optuna configuration for hyperparameter optimization.

    Flattened from plugin architecture to top-level for easier access.
    Replaces: settings.SESSION.training.plugins["optuna"]
    New usage: settings.OPTUNA

    Args:
        enabled: Whether Optuna optimization is enabled
        n_trials: Number of optimization trials to run
        direction: Optimization direction (minimize or maximize)
        study_name: Optional study name for persistence and reuse
        storage: Optional storage URL (e.g., sqlite:///optuna.db)
        sampler: Sampler configuration for hyperparameter suggestions
        pruner: Pruner configuration for early stopping
        model: Hierarchical model parameter ranges that mirror MODEL structure
    """

    enabled: bool = Field(default=False, description="Whether to enable Optuna optimization")
    n_trials: PositiveInt = Field(
        default=3, description="Number of trials for hyperparameter optimization"
    )
    direction: Literal["minimize", "maximize"] = Field(
        default="minimize", description="Optimization direction"
    )
    study_name: str | None = Field(default=None, description="Optuna study name")
    storage: str | None = Field(default=None, description="Optuna storage URL")
    sampler: SamplerSettings = Field(
        default_factory=SamplerSettings, description="Optuna sampler configuration"
    )
    pruner: PrunerSettings = Field(
        default_factory=PrunerSettings, description="Optuna pruner configuration"
    )
    model: dict[str, Any] = Field(
        default_factory=dict,
        description="Hierarchical model parameter ranges that mirror MODEL structure",
    )

    # Validation now handled by proper Pydantic types:
    # - PositiveInt for n_trials ensures positive values
    # - Literal["minimize", "maximize"] for direction ensures valid values
    # - sampler/pruner validation happens in ComponentSettings

    @property
    def has_model_ranges(self) -> bool:
        """Check if model parameter ranges are specified for optimization.

        Returns:
            bool: True if model parameter ranges are configured
        """
        return len(self.model) > 0

    # Sampler/pruner config should be consumed via `.sampler.to_dict()` and
    # `.pruner.to_dict()` directly. Explicit helper methods are removed to
    # ensure consistent usage of settings serialization across the codebase.
