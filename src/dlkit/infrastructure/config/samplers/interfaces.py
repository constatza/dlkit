"""Settings sampling interfaces following Interface Segregation Principle."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from dlkit.infrastructure.config.workflow_configs import OptimizationWorkflowConfig


@runtime_checkable
class OptunaTrialProtocol(Protocol):
    """Structural protocol matching optuna.trial.Trial's suggest methods."""

    def suggest_float(
        self, name: str, low: float, high: float, *, log: bool = False, step: float | None = None
    ) -> float: ...

    def suggest_int(
        self, name: str, low: int, high: int, *, log: bool = False, step: int = 1
    ) -> int: ...

    def suggest_categorical(
        self, name: str, choices: Sequence[str | int | float | bool]
    ) -> str | int | float | bool: ...

    @property
    def params(self) -> dict[str, str | int | float | bool]: ...


@runtime_checkable
class ISettingsSampler(Protocol):
    """Interface for hyperparameter sampling following ISP.

    Single responsibility: Convert trial suggestions into complete settings
    ready for optimization workflows.
    """

    def sample(
        self, trial: OptunaTrialProtocol, base_settings: OptimizationWorkflowConfig
    ) -> OptimizationWorkflowConfig:
        """Sample hyperparameters and return complete settings.

        Args:
            trial: Optimization trial object (e.g., Optuna trial)
            base_settings: Base optimization workflow configuration with concrete default values

        Returns:
            Settings with sampled hyperparameters applied
        """
        ...


class NullSettingsSampler(ISettingsSampler):
    """Null Object pattern - returns settings unchanged.

    Used when no hyperparameter optimization is configured.
    Follows SOLID principles by providing same interface.
    """

    def sample(
        self, trial: OptunaTrialProtocol, base_settings: OptimizationWorkflowConfig
    ) -> OptimizationWorkflowConfig:
        """Return base settings unchanged.

        Args:
            trial: Optimization trial object (unused)
            base_settings: Base configuration settings

        Returns:
            Unchanged base settings
        """
        return base_settings
