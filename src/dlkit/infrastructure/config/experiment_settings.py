"""Experiment settings — MLflow experiment identity."""

from __future__ import annotations

from pydantic import Field

from dlkit.infrastructure.config.core.base_settings import BasicSettings


class ExperimentSettings(BasicSettings):
    """MLflow experiment identity.

    Args:
        name: Experiment name used in the tracking backend.
        run_name: Optional run name override.
        tags: Key-value tags attached to every run.
    """

    name: str = "dlkit-experiment"
    run_name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
