"""Optimization abstractions following DIP."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from dlkit.tools.config import GeneralSettings


class IOptimizationResult(ABC):
    """Abstract optimization result."""

    @property
    @abstractmethod
    def best_params(self) -> dict[str, Any]:
        """Best hyperparameters found."""
        raise NotImplementedError

    @property
    @abstractmethod
    def best_value(self) -> float:
        """Best objective value achieved."""
        raise NotImplementedError

    @property
    @abstractmethod
    def trial_number(self) -> int:
        """Best trial number."""
        raise NotImplementedError

    @property
    @abstractmethod
    def study_summary(self) -> dict[str, Any]:
        """Summary of the optimization study."""
        raise NotImplementedError


class IHyperparameterOptimizer(ABC):
    """Abstract hyperparameter optimizer following DIP."""

    @abstractmethod
    def optimize(
        self,
        objective: Callable[[Any], float],
        settings: GeneralSettings,
        n_trials: int,
        direction: str = "minimize",
    ) -> IOptimizationResult:
        """Run hyperparameter optimization.

        Args:
            objective: Objective function to optimize
            settings: Base settings for sampling
            n_trials: Number of optimization trials
            direction: Optimization direction ("minimize" or "maximize")

        Returns:
            Optimization result with best parameters and metadata
        """
        raise NotImplementedError

    @abstractmethod
    def create_sampled_settings(
        self, base_settings: GeneralSettings, trial: Any
    ) -> GeneralSettings:
        """Create settings with sampled hyperparameters.

        Args:
            base_settings: Base configuration settings
            trial: Optimization trial object for sampling

        Returns:
            Settings with sampled hyperparameters
        """
        raise NotImplementedError
