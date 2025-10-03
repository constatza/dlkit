"""Core settings base classes with SOLID principles."""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

from pydantic_settings import BaseSettings, SettingsConfigDict


class BasicSettings(BaseSettings):
    """Base settings class with validation and serialization.

    This is the foundation for all settings classes in DLKit.
    It provides consistent validation, serialization, and compatibility checking.
    """

    model_config = SettingsConfigDict(
        frozen=True,
        validate_default=True,
        validate_by_alias=True,
        validate_by_name=True,
        validate_assignment=True,
        nested_model_default_partial_update=True,
        case_sensitive=True,
        extra="forbid",
        # Fix pydantic-settings bug where model_validate is protected
        protected_namespaces=("settings_customise_sources",),
    )

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Serialize settings to a plain dict.

        Excludes values that are None or not explicitly set, to reduce
        noisy kwargs passed to downstream constructors.

        Args:
            exclude: Additional keys to exclude from the resulting dict

        Returns:
            A clean dictionary representation of the settings
        """
        extra = set(exclude or set())
        # Exclude None values, include defaults/unset so downstream receives
        # the full, explicit configuration. Extras are preserved.
        return self.model_dump(exclude_none=True, exclude=extra)

    # Deliberately no "to_dict_compatible_with": construction belongs in factories.


class ComponentSettings[T](BasicSettings):
    """Settings for components that can be dynamically constructed.

    This replaces the old ClassSettings but removes the build() method
    to follow SOLID principles. Object construction is now handled by factories.

    Args:
        name: Name/class reference for the component
        module_path: Module path for dynamic imports
    """

    model_config = SettingsConfigDict(extra="allow", arbitrary_types_allowed=True)
    # Use Any for base types to allow subclasses to be more specific without LSP violations
    # Subclasses can override with specific types that are compatible
    name: Any
    module_path: Any = None

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Serialize component settings, excluding meta fields.

        Always excludes component selector fields ("name", "module_path").
        """
        extra = {"name", "module_path"}
        if exclude:
            extra.update(exclude)
        return super().to_dict(exclude=extra)

    def get_init_kwargs(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Get initialization kwargs for component construction.

        Args:
            exclude: Additional keys to exclude beyond defaults

        Returns:
            dict[str, Any]: Initialization kwargs for the component
        """
        default_exclude = {"name", "module_path"}
        if exclude:
            default_exclude.update(exclude)

        return self.model_dump(
            exclude=default_exclude,
            exclude_none=True,
            exclude_unset=True,
        )


class HyperParameterSettings(BaseSettings):
    """Settings that can contain hyperparameter specifications.

    This class only holds hyperparameter configuration
    For sampling operations, use SettingsSampler classes which follow SRP.

    Note: The sample() and get_optuna_suggestion() methods have been moved to
    SettingsSampler implementations to maintain Single Responsibility Principle.
    Settings classes should only hold and validate configuration
    """

    pass

    @staticmethod
    def deep_merge_model(base_model: Any, sampled_params: dict[str, Any]) -> dict[str, Any]:
        """Deep merge sampled hyperparameters into base model settings.

        Args:
            base_model: Base model settings object
            sampled_params: Dictionary of sampled parameter values with dot-notation keys

        Returns:
            dict: Updated model dictionary for model_copy

        Example:
            base_model.hidden_size = 64
            sampled_params = {"hidden_size": 128, "optimizer.lr": 0.001}
            -> {"hidden_size": 128, "optimizer": {"lr": 0.001}}
        """
        # Convert base model to dict
        if hasattr(base_model, "model_dump"):
            result = base_model.model_dump()
        elif hasattr(base_model, "dict"):
            result = base_model.dict()
        else:
            result = dict(base_model) if hasattr(base_model, "__dict__") else {}

        # Deep merge sampled parameters
        for param_path, value in sampled_params.items():
            HyperParameterSettings._set_nested_value(result, param_path, value)

        return result

    @staticmethod
    def _set_nested_value(target_dict: dict, path: str, value: Any) -> None:
        """Set a nested value in a dictionary using dot notation path.

        Args:
            target_dict: Target dictionary to update
            path: Dot-notation path (e.g., "optimizer.lr")
            value: Value to set
        """
        keys = path.split(".")
        current = target_dict

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value
