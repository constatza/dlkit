"""Core settings base classes with SOLID principles.

Architecture: Immutable Settings with Functional Patching
---------------------------------------------------------
DLKit uses frozen Pydantic settings (``frozen=True``) for all config classes.
Updates produce new instances via :func:`~dlkit.tools.config.core.patching.patch_model`
rather than mutating existing ones.  This gives:

- Referential transparency: passed-around config objects cannot be silently changed.
- Explicit dataflow: callers must capture the returned new instance.
- Full type safety: patch values are validated against field annotations before use.

To update a config::

    from dlkit.tools.config.core.patching import patch_model

    new_cfg = patch_model(cfg, {"TRAINING": {"epochs": 100}})
    new_cfg = cfg.patch({"TRAINING.optimizer.lr": 0.001})  # dotted-key shorthand

Note: ``exclude=True`` fields (e.g. ``ValueBasedEntry.value``) are preserved by
``model_copy()`` because it copies ``__dict__`` directly — no serialisation round-trip.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Self

from pydantic_settings import BaseSettings, SettingsConfigDict

from dlkit.tools.config.core.patching import patch_model
from dlkit.tools.config.core.updater import update_settings


def _auto_section_name(cls: type) -> str:
    """Derive TOML section name from settings class name.

    Args:
        cls: Settings class to derive name from.

    Returns:
        Uppercase section name, e.g. ``SessionSettings`` → ``"SESSION"``.
    """
    name = cls.__name__
    return name[:-8].upper() if name.endswith("Settings") else name.upper()


class BasicSettings(BaseSettings):
    """Base settings class with validation and serialization.

    This is the foundation for all settings classes in DLKit.
    It provides consistent validation, serialization, and compatibility checking.
    All instances are immutable (``frozen=True``); use :meth:`patch` to derive
    new instances with changed fields.
    """

    model_config = SettingsConfigDict(
        frozen=True,
        validate_default=True,
        validate_by_alias=True,
        validate_by_name=True,
        case_sensitive=True,
        extra="forbid",
        # Fix pydantic-settings bug where model_validate is protected
        protected_namespaces=("settings_customise_sources",),
        # Suppress serialization warnings when nested models are dicts
        ser_json_timedelta="float",
        ser_json_bytes="base64",
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
        return self.model_dump(exclude_none=True, exclude=extra)

    def update_with(self, updates: dict[str, Any], *, validate: bool = True) -> Self:
        """Return a new settings instance with *updates* applied (deep merge).

        Convenience wrapper around :func:`~dlkit.tools.config.core.patching.patch_model`.
        The current instance is **not** mutated.

        Args:
            updates: Nested mapping of fields to update.  Dotted keys are
                expanded (e.g. ``"a.b"`` → ``{"a": {"b": ...}}``).
            validate: Passed to ``patch_model`` as ``revalidate``.

        Returns:
            New settings instance of the same concrete type.
        """
        return update_settings(self, updates, validate=validate)

    def patch(
        self,
        overrides: Mapping[str, Any],
        *,
        sep: str = ".",
        revalidate: bool = True,
    ) -> Self:
        """Return a new instance with *overrides* applied.

        Supports dotted keys (``"a.b.c"``) and plain nested dicts.
        Collisions in the override set raise ``ValueError`` (strict merge).

        Args:
            overrides: Mixed overrides mapping.
            sep: Dotted-path separator (default ``"."``).
            revalidate: If ``True``, performs a full validation pass on the
                result model (recommended; default ``True``).

        Returns:
            New settings instance of the same concrete type.

        Raises:
            ValueError: On key conflicts in overrides.
            KeyError: Unknown field name.
            pydantic.ValidationError: On type mismatches.
        """
        return patch_model(self, overrides, sep=sep, revalidate=revalidate)

    @classmethod
    def from_toml(cls, config_path: Path | str, **overrides: Any) -> Self:
        """Load a single settings section from a TOML file.

        Infers section name from the class name (``SessionSettings`` → ``SESSION``).
        Priority: TOML values < env vars (``DLKIT_<SECTION>__<field>``) < ``**overrides``.

        Args:
            config_path: Path to the TOML configuration file.
            **overrides: Field overrides applied last (highest priority).

        Returns:
            Validated settings instance of the calling class.

        Raises:
            FileNotFoundError: If the config file does not exist.
            pydantic.ValidationError: If validation fails.
        """
        from dlkit.tools.config.core.sources import DLKitTomlSource, _read_env_patches

        section_name = _auto_section_name(cls)
        source = DLKitTomlSource(Path(config_path), sections=[section_name])
        section_data = source().get(section_name, {})

        settings = cls.model_validate(section_data)
        if env := _read_env_patches(f"DLKIT_{section_name}__", "__"):
            settings = patch_model(settings, env)
        if overrides:
            settings = patch_model(settings, overrides)
        return settings


class ComponentSettings(BasicSettings):
    """Settings for components that can be dynamically constructed.

    This replaces the old ClassSettings but removes the build() method
    to follow SOLID principles. Object construction is now handled by factories.
    Subclasses can declare their own name/module_path fields as needed.
    """

    model_config = SettingsConfigDict(extra="allow", arbitrary_types_allowed=True)

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


class HyperParameterSettings(BasicSettings):
    """Settings that can contain hyperparameter specifications.

    This class only holds hyperparameter configuration
    For sampling operations, use SettingsSampler classes which follow SRP.

    Note: The sample() and get_optuna_suggestion() methods have been moved to
    SettingsSampler implementations to maintain Single Responsibility Principle.
    Settings classes should only hold and validate configuration
    """

    pass
