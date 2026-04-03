"""Type-safe configuration accessor to replace getattr chains.

Single Responsibility: Provide typed access to nested configuration values with fallbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.workflow_configs import OptimizationWorkflowConfig, TrainingWorkflowConfig

type _WorkflowSettings = GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig


class ConfigAccessor:
    """Type-safe access to configuration settings.

    Eliminates primitive obsession by replacing getattr chains with typed methods.
    Provides defaults and safe navigation through nested settings.

    Args:
        settings: Global configuration settings
    """

    def __init__(self, settings: _WorkflowSettings):
        """Initialize with global settings.

        Args:
            settings: Global configuration settings
        """
        self._settings = settings

    def get_model_name(self) -> str:
        """Get model name with fallback.

        Returns:
            Model name from settings or default "Model"
        """
        return self._get_nested("MODEL", "name", default="Model")

    def get_session_root_dir(self) -> Path | None:
        """Get session root directory.

        Returns:
            Root directory path or None if not configured
        """
        return self._get_nested("SESSION", "root_dir", default=None)

    def get_mlflow_config(self) -> Any:
        """Get MLflow configuration section.

        Returns:
            MLflow configuration object or None
        """
        return getattr(self._settings, "MLFLOW", None)

    def get_mlflow_client_config(self) -> Any:
        """Get MLflow configuration object (flat schema)."""
        return self.get_mlflow_config()

    def get_run_name(self) -> str | None:
        """Get configured MLflow run name.

        Returns:
            Run name or None if not configured
        """
        client = self.get_mlflow_client_config()
        return getattr(client, "run_name", None) if client else None

    def get_registered_model_name(self) -> str | None:
        """Get optional registered model name override."""
        mlflow_cfg = self.get_mlflow_client_config()
        return getattr(mlflow_cfg, "registered_model_name", None) if mlflow_cfg else None

    def get_registered_model_aliases(self) -> tuple[str, ...] | None:
        """Get optional registered model aliases override."""
        mlflow_cfg = self.get_mlflow_client_config()
        aliases = getattr(mlflow_cfg, "registered_model_aliases", None) if mlflow_cfg else None
        if not aliases:
            return None
        normalized = tuple(str(alias).strip() for alias in aliases if str(alias).strip())
        return normalized if normalized else None

    def get_registered_model_version_tags(self) -> dict[str, str]:
        """Get optional registered model version tag overrides."""
        mlflow_cfg = self.get_mlflow_client_config()
        tags = getattr(mlflow_cfg, "registered_model_version_tags", None) if mlflow_cfg else None
        if not isinstance(tags, dict):
            return {}
        return {str(key).strip(): str(value) for key, value in tags.items() if str(key).strip()}

    def should_register_model(self) -> bool:
        """Check if model registration is enabled.

        Returns:
            True if model registration is enabled
        """
        mlflow_cfg = self.get_mlflow_client_config()
        return bool(mlflow_cfg and getattr(mlflow_cfg, "register_model", False))

    def get_extras(self) -> Any | None:
        """Get EXTRAS configuration section.

        Returns:
            EXTRAS configuration object or None
        """
        return getattr(self._settings, "EXTRAS", None)

    def get_mlflow_params(self) -> dict[str, Any]:
        """Get user-defined MLflow parameters from EXTRAS.

        Returns:
            Dictionary of parameters or empty dict
        """
        extras = self.get_extras()
        if not extras:
            return {}

        params = getattr(extras, "mlflow_params", None)
        return params if isinstance(params, dict) else {}

    def get_mlflow_artifacts(self) -> list[str]:
        """Get user-defined artifact paths from EXTRAS.

        Returns:
            List of artifact paths
        """
        extras = self.get_extras()
        if not extras:
            return []

        artifacts = getattr(extras, "mlflow_artifacts", None)
        if not artifacts:
            return []

        # Normalize to list
        if isinstance(artifacts, (list, tuple)):
            return list(artifacts)
        return [artifacts]

    def get_run_tags(self) -> dict[str, str] | None:
        """Get run tags from MLflow configuration.

        Returns:
            Tags dict or None if not configured
        """
        mlflow_cfg = self.get_mlflow_client_config()
        return getattr(mlflow_cfg, "tags", None) if mlflow_cfg else None

    def get_mlflow_artifacts_toml(self) -> dict[str, dict[str, Any]]:
        """Get user-defined TOML artifacts from EXTRAS.

        Returns:
            Dictionary mapping artifact names to data dicts
        """
        extras = self.get_extras()
        if not extras:
            return {}

        artifacts = getattr(extras, "mlflow_artifacts_toml", None)
        return artifacts if isinstance(artifacts, dict) else {}

    def _get_nested(self, *keys: str, default: Any = None) -> Any:
        """Safe nested attribute access with default.

        Args:
            *keys: Sequence of attribute names to traverse
            default: Default value if any attribute is missing

        Returns:
            Nested attribute value or default

        Example:
            >>> accessor._get_nested("MODEL", "name", default="default")
            # Equivalent to: getattr(getattr(settings, "MODEL", None), "name", "default")
        """
        obj = self._settings
        for key in keys:
            if obj is None:
                return default
            obj = getattr(obj, key, None)

        return obj if obj is not None else default
