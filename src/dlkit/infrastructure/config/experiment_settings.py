"""Experiment settings — MLflow experiment identity and model registration metadata."""

from __future__ import annotations

from pydantic import Field

from dlkit.infrastructure.config.core.base_settings import BasicSettings


class ExperimentSettings(BasicSettings):
    """MLflow experiment identity and model registration metadata.

    Args:
        name: Experiment name used in the tracking backend.
        run_name: Optional run name override.
        tags: Key-value tags attached to every run.
        register_model: Whether to register model artifacts after training.
        registered_model_name: Override name for the registered model.
        registered_model_aliases: Aliases to attach after model registration.
        registered_model_version_tags: Tags for registered model versions.
    """

    name: str = "dlkit-experiment"
    run_name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    register_model: bool = False
    registered_model_name: str | None = None
    registered_model_aliases: tuple[str, ...] | None = None
    registered_model_version_tags: dict[str, str] | None = None
