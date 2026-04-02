"""Section mapping registry for TOML config sections."""

from typing import cast

from pydantic import BaseModel

from dlkit.tools.io.config_errors import ConfigSectionError

# Section mapping registry: Maps between Pydantic model classes and TOML section names
_SECTION_MAPPING: dict[type[BaseModel], str] = {}
_SECTION_NAME_MAPPING: dict[str, type[BaseModel]] = {}
_DEFAULT_SECTION_NAME_MAPPING: dict[str, type[BaseModel]] = {}


def _apply_mapping(model_class: type[BaseModel], section_name: str) -> None:
    """Apply a bidirectional mapping between model class and section name."""
    _SECTION_MAPPING[model_class] = section_name
    _SECTION_NAME_MAPPING[section_name] = model_class


def register_section_mapping(model_class: type[BaseModel], section_name: str) -> None:
    """Register a mapping between a Pydantic model class and TOML section name.

    Args:
        model_class: The Pydantic model class
        section_name: The corresponding TOML section name (case-sensitive)
    """
    normalized = section_name.upper()
    _apply_mapping(model_class, normalized)


def get_section_name(model_class: type[BaseModel]) -> str:
    """Get the TOML section name for a Pydantic model class.

    Args:
        model_class: The Pydantic model class

    Returns:
        The TOML section name

    Raises:
        ConfigSectionError: If no mapping is found and auto-detection fails
    """
    # Check explicit mapping first
    if model_class in _SECTION_MAPPING:
        return _SECTION_MAPPING[model_class]

    # Auto-detect from class name: remove "Settings" suffix and uppercase
    class_name = model_class.__name__
    if class_name.endswith("Settings"):
        base_name = class_name[:-8]  # Remove "Settings"
        return base_name.upper()

    # For special cases, try direct uppercase
    section = class_name.upper()
    _apply_mapping(model_class, section)
    return section


def get_model_class_for_section(section_name: str) -> type[BaseModel]:
    """Lookup the Pydantic model class registered for a TOML section name."""
    normalized = section_name.upper()
    model_class = _SECTION_NAME_MAPPING.get(normalized)
    if model_class is None:
        raise ConfigSectionError(
            f"No settings model registered for section '{section_name}'.",
            section_name=section_name,
            available_sections=list(_SECTION_NAME_MAPPING.keys()),
        )
    return model_class


def _resolve_section_models(
    section_configs: dict[str, type[BaseModel] | None] | list[str],
) -> dict[str, type[BaseModel]]:
    """Resolve mapping of section names to model classes using registry defaults."""
    from collections.abc import Mapping, Sequence

    if isinstance(section_configs, Mapping):
        items = list(section_configs.items())
    elif isinstance(section_configs, Sequence) and not isinstance(section_configs, (str, bytes)):
        items = [(name, None) for name in section_configs]
    else:
        raise TypeError(
            "section_configs must be a mapping of section->model or a sequence of section names"
        )

    resolved: dict[str, type[BaseModel]] = {}
    for raw_name, model_class in items:
        if not isinstance(raw_name, str):
            raise TypeError("Section names must be strings")
        section_name = raw_name.upper()
        resolved_model = model_class or _SECTION_NAME_MAPPING.get(section_name)
        if resolved_model is None:
            raise ConfigSectionError(
                f"No registered model for section '{raw_name}'. Provide a model_class or register one.",
                section_name=raw_name,
                available_sections=list(_SECTION_NAME_MAPPING.keys()),
            )
        resolved[section_name] = cast("type[BaseModel]", resolved_model)
    return resolved


def reset_section_mappings(section_name: str | None = None) -> None:
    """Reset section mappings to their defaults.

    Args:
        section_name: Optional section name to reset; resets all when omitted.
    """
    if section_name is not None:
        normalized = section_name.upper()
        default_model = _DEFAULT_SECTION_NAME_MAPPING.get(normalized)
        # Remove any existing mappings for this section
        for model, mapped_section in list(_SECTION_MAPPING.items()):
            if mapped_section == normalized:
                _SECTION_MAPPING.pop(model, None)
        if default_model is not None:
            _apply_mapping(default_model, normalized)
        else:
            _SECTION_NAME_MAPPING.pop(normalized, None)
        return

    _SECTION_MAPPING.clear()
    _SECTION_NAME_MAPPING.clear()
    for section, model_cls in _DEFAULT_SECTION_NAME_MAPPING.items():
        _apply_mapping(model_cls, section)


def _initialize_default_mappings() -> None:
    """Initialize default section mappings for common settings classes."""
    try:
        from dlkit.tools.config.datamodule_settings import DataModuleSettings
        from dlkit.tools.config.dataset_settings import DatasetSettings
        from dlkit.tools.config.extras_settings import ExtrasSettings
        from dlkit.tools.config.model_components import ModelComponentSettings
        from dlkit.tools.config.optuna_settings import OptunaSettings
        from dlkit.tools.config.paths_settings import PathsSettings
        from dlkit.tools.config.session_settings import SessionSettings
        from dlkit.tools.config.training_settings import TrainingSettings

        default_pairs: tuple[tuple[str, type[BaseModel]], ...] = (
            ("SESSION", SessionSettings),
            ("MODEL", ModelComponentSettings),
            ("DATAMODULE", DataModuleSettings),
            ("DATASET", DatasetSettings),
            ("TRAINING", TrainingSettings),
            ("OPTUNA", OptunaSettings),
            ("PATHS", PathsSettings),
            ("EXTRAS", ExtrasSettings),
        )

        for section, model_cls in default_pairs:
            normalized = section.upper()
            _DEFAULT_SECTION_NAME_MAPPING.setdefault(normalized, model_cls)
            # Apply defaults without overwriting explicit registrations that may
            # have happened earlier in the lifecycle.
            if normalized not in _SECTION_NAME_MAPPING:
                _apply_mapping(model_cls, normalized)
    except Exception:
        # Default mappings are best-effort; silently ignore import errors to avoid
        # circular import issues during early module import.
        pass


# Initialize default mappings
_initialize_default_mappings()
