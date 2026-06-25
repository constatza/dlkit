"""Compatibility shim: backward-compatible MLflowSettings stub.

Tests and engine code that have not yet migrated to the new ExperimentSettings +
TrackingSettings split still use MLflowSettings directly. This shim keeps them working.
Will be removed in Task 5 when all callers are updated.
"""

from __future__ import annotations

from dlkit.infrastructure.config.core.base_settings import BasicSettings


class MLflowSettings(BasicSettings):
    """Backward-compatible MLflow settings.

    Merges fields from the old MLflowSettings into a single class.
    In the new architecture these are split across ExperimentSettings (experiment_name,
    run_name, tags, register_model, ...) and TrackingSettings (uri, backend, max_retries).

    Args:
        experiment_name: MLflow experiment name.
        run_name: Optional run name.
        tags: Key-value tags for the run.
        register_model: Whether to register the model after training.
        registered_model_name: Override name for registered model.
        registered_model_aliases: Aliases to attach after registration.
        registered_model_version_tags: Tags for registered model versions.
        uri: MLflow tracking URI.
        backend: Tracking backend type.
        max_retries: Maximum connection retries.
    """

    experiment_name: str = "dlkit-experiment"
    run_name: str | None = None
    tags: dict[str, str] = {}
    register_model: bool = False
    registered_model_name: str | None = None
    registered_model_aliases: tuple[str, ...] | None = None
    registered_model_version_tags: dict[str, str] | None = None

    # Connection settings (from TrackingSettings)
    uri: str | None = None
    backend: str = "mlflow"
    max_retries: int = 3


__all__ = ["MLflowSettings"]
