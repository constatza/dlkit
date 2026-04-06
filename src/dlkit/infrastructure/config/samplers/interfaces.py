"""Settings sampling interfaces following Interface Segregation Principle."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from dlkit.infrastructure.config import GeneralSettings


@runtime_checkable
class ISettingsSampler(Protocol):
    """Interface for hyperparameter sampling following ISP.

    Single responsibility: Convert trial suggestions into complete GeneralSettings
    ready for any workflow (training, inference, etc.).
    """

    def sample(self, trial: Any, base_settings: GeneralSettings) -> GeneralSettings:
        """Sample hyperparameters and return complete settings.

        Args:
            trial: Optimization trial object (e.g., Optuna trial)
            base_settings: Base configuration with concrete default values

        Returns:
            GeneralSettings with sampled hyperparameters applied
        """
        ...


class NullSettingsSampler:
    """Null Object pattern - returns settings unchanged.

    Used when no hyperparameter optimization is configured.
    Follows SOLID principles by providing same interface.
    """

    def sample(self, trial: Any, base_settings: GeneralSettings) -> GeneralSettings:
        """Return base settings unchanged."""
        return base_settings
