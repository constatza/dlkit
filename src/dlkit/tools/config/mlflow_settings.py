"""MLflow settings using a flat client-only configuration."""

from __future__ import annotations

from pydantic import Field, model_validator

from dlkit.tools.utils.logging_config import get_logger

from .core.base_settings import BasicSettings

logger = get_logger(__name__)


class MLflowSettings(BasicSettings):
    """MLflow tracking and registry configuration.

    Infrastructure endpoints are intentionally env-driven:
    - ``MLFLOW_TRACKING_URI``
    - ``MLFLOW_ARTIFACT_URI``

    Note: The presence of an ``[MLFLOW]`` config section enables tracking.
    There is no separate ``enabled`` flag.
    """

    experiment_name: str = Field(default="Experiment", description="MLflow experiment name")
    run_name: str | None = Field(default=None, description="Optional MLflow run name")
    tags: dict[str, str] | None = Field(
        default=None,
        description="Key-value tags attached to every MLflow run",
    )
    register_model: bool = Field(default=False, description="Whether to register model artifacts")
    registered_model_name: str | None = Field(
        default=None,
        description="Optional override for registered model name",
    )
    registered_model_aliases: tuple[str, ...] | None = Field(
        default=None,
        description="Optional aliases to attach after model registration",
    )
    registered_model_version_tags: dict[str, str] | None = Field(
        default=None,
        description="Optional tags to attach to registered model versions",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum connection retries for transient MLflow client operations.",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_and_infra_toml_fields(cls, data: object) -> object:
        """Fail fast on removed nested sections and env-only infrastructure keys."""
        if not isinstance(data, dict):
            return data

        legacy_sections = [key for key in ("server", "client") if key in data]
        if legacy_sections:
            joined = ", ".join(legacy_sections)
            raise ValueError(
                "Legacy MLflow config sections are no longer supported: "
                f"{joined}. Use flat [MLFLOW] fields and env vars "
                "(MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_URI)."
            )

        infra_fields = [key for key in ("tracking_uri", "artifacts_destination") if key in data]
        if infra_fields:
            joined = ", ".join(infra_fields)
            raise ValueError(
                "MLflow infrastructure fields are env-only and must not be set in TOML: "
                f"{joined}. Use MLFLOW_TRACKING_URI and MLFLOW_ARTIFACT_URI."
            )

        if "enabled" in data:
            raise ValueError(
                "MLflowSettings no longer has an 'enabled' field. "
                "The presence of the [MLFLOW] section in config enables tracking."
            )

        return data

    @model_validator(mode="after")
    def warn_missing_model_name(self) -> MLflowSettings:
        """Warn when model registration is enabled but no name is configured."""
        if self.register_model and not self.registered_model_name:
            logger.warning(
                "MLflow model registration is enabled without 'registered_model_name'; "
                "falling back to the model class name."
            )
        return self
