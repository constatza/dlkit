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

    def get_registered_model_name(self) -> str | None:
        """Return optional registered model name."""
        exp = self._settings.experiment
        return exp.registered_model_name if exp else None

    def get_registered_model_aliases(self) -> tuple[str, ...] | None:
        """Return normalised model aliases, or None if absent."""
        exp = self._settings.experiment
        aliases = exp.registered_model_aliases if exp else None
        if not aliases:
            return None
        normalised = tuple(str(a).strip() for a in aliases if str(a).strip())
        return normalised or None

    def get_registered_model_version_tags(self) -> dict[str, str]:
        """Return model version tags, empty dict if absent."""
        exp = self._settings.experiment
        tags = exp.registered_model_version_tags if exp else None
        if not tags:
            return {}
        return {str(k).strip(): str(v) for k, v in tags.items() if str(k).strip()}

    def should_register_model(self) -> bool:
        """Return True if model registration is enabled."""
        exp = self._settings.experiment
        return bool(exp and exp.register_model)

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
