"""Type-safe configuration accessor for experiment tracking."""

from __future__ import annotations

from dlkit.infrastructure.config.experiment_settings import ExperimentSettings
from dlkit.infrastructure.config.job_config import JobConfig


class ConfigAccessor:
    """Typed access to JobConfig fields needed by the tracking layer.

    Args:
        settings: A JobConfig instance.
    """

    def __init__(self, settings: JobConfig) -> None:
        self._settings = settings

    def get_model_name(self) -> str:
        """Get model name or default."""
        name = self._settings.model.name if self._settings.model else None
        if name is None:
            return "Model"
        return name.__qualname__ if isinstance(name, type) else str(name)

    def get_mlflow_config(self) -> ExperimentSettings | None:
        """Return the experiment settings section (MLflow identity fields)."""
        return self._settings.experiment

    def get_run_name(self) -> str | None:
        """Return configured MLflow run name."""
        exp = self._settings.experiment
        return exp.run_name if exp else None

    def get_run_tags(self) -> dict[str, str] | None:
        """Return run tags from experiment settings."""
        exp = self._settings.experiment
        return exp.tags if exp else None

    def get_extras(self) -> None:
        """Return extras config section (always None in new schema)."""
        return None

    def get_mlflow_params(self) -> dict[str, str]:
        """Return user-defined MLflow parameters (always empty in new schema)."""
        return {}

    def get_mlflow_artifacts(self) -> list[str]:
        """Return user-defined artifact paths (always empty in new schema)."""
        return []

    def get_mlflow_artifacts_toml(self) -> dict[str, dict[str, str]]:
        """Return TOML artifact definitions (always empty in new schema)."""
        return {}
