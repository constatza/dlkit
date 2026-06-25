"""Compatibility shim: minimal OptunaSettings for callers not yet migrated to SearchSettings.

Will be removed in Task 5 when all callers are updated.
"""

from __future__ import annotations

from typing import Any

from dlkit.infrastructure.config.core.base_settings import BasicSettings
from dlkit.infrastructure.config.search_settings import (
    PrunerSettings,
    SamplerSettings,
    SearchSettings,
)


class OptunaSettings(BasicSettings):
    """Minimal Optuna settings stub for backward compatibility.

    Args:
        enabled: Whether Optuna optimization is enabled.
        n_trials: Number of optimization trials.
        direction: Optimization direction.
        study_name: Optional study name.
        storage: Optional Optuna storage URL.
        model: Free-form hyperparameter search space (old-style dict).
    """

    enabled: bool = False
    n_trials: int = 10
    direction: str = "minimize"
    study_name: str | None = None
    storage: str | None = None
    model: dict[str, Any] = {}

    @property
    def has_model_ranges(self) -> bool:
        """Return True when the model search space is non-empty."""
        return bool(self.model)

    @classmethod
    def from_search_settings(cls, search: SearchSettings) -> OptunaSettings:
        """Convert new SearchSettings to legacy OptunaSettings.

        Args:
            search: New-style search configuration.

        Returns:
            Equivalent OptunaSettings instance.
        """
        return cls(
            enabled=bool(search.space),
            n_trials=search.n_trials,
            direction=search.direction,
            study_name=search.study_name,
            storage=search.storage,
        )


__all__ = ["OptunaSettings", "PrunerSettings", "SamplerSettings"]
