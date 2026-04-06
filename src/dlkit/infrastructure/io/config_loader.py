"""TOML config loading functions."""

import sys
import tomllib
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, overload

from pydantic import BaseModel

from dlkit.infrastructure.io.config_errors import ConfigSectionError, ConfigValidationError
from dlkit.infrastructure.io.config_section_registry import (
    _resolve_section_models,
    get_model_class_for_section,
    get_section_name,
)

if TYPE_CHECKING:
    from dlkit.infrastructure.config.workflow_configs import (
        InferenceWorkflowConfig,
        OptimizationWorkflowConfig,
        TrainingWorkflowConfig,
    )


def _sync_session_root_to_environment(settings: Any) -> None:
    """Synchronize SESSION.root_dir to global EnvironmentSettings if appropriate.

    This provides a defensive fallback when PathOverrideContext is not active.
    Respects precedence: DLKIT_ROOT_DIR env var > SESSION.root_dir > CWD.

    Args:
        settings: Loaded settings object (GeneralSettings or similar)
    """
    try:
        import os

        from loguru import logger

        from dlkit.infrastructure.config.environment import env as global_environment
        from dlkit.infrastructure.io.paths import coerce_root_dir_to_absolute

        # Only update if EnvironmentSettings doesn't already have root_dir from env var
        if os.environ.get("DLKIT_ROOT_DIR"):
            # Explicit env var takes precedence - don't override
            return

        # Extract SESSION.root_dir if present
        session = getattr(settings, "SESSION", None)
        if session is None:
            return

        session_root_dir = getattr(session, "root_dir", None)
        if session_root_dir is None:
            return

        normalized_root = coerce_root_dir_to_absolute(session_root_dir)
        if normalized_root is None:
            logger.debug(
                "SESSION.root_dir is not absolute; skipping environment sync",
                session_root_dir=str(session_root_dir),
            )
            return

        # Update global environment for fallback resolution
        # This ensures SESSION.root_dir is respected even when PathOverrideContext is not active
        global_environment.root_dir = str(normalized_root)

        logger.debug(
            "Synchronized SESSION.root_dir to EnvironmentSettings for fallback path resolution",
            session_root_dir=str(normalized_root),
        )
    except Exception as e:
        # Non-fatal - path resolution will fall back to CWD if this fails
        from loguru import logger

        logger.warning(f"Failed to sync SESSION.root_dir to environment (non-fatal): {e}")


def _resolve_default_settings_class() -> type[BaseModel] | None:
    """Lazily import GeneralSettings without top-level coupling."""
    module_name = "dlkit.infrastructure.config.general_settings"
    module = sys.modules.get(module_name)

    if module is None:
        if find_spec(module_name) is None:
            return None
        module = import_module(module_name)

    general = getattr(module, "GeneralSettings", None)
    if isinstance(general, type) and issubclass(general, BaseModel):
        return general
    return None


def load_config[T: BaseModel](
    config_path: Path | str,
    model_class: type[T] | None = None,
    raw: bool = False,
) -> T | dict[str, Any]:
    """Load TOML config file using dynaconf.

    By default, loads the config as raw dict. Use model_class to specify validation.

    Args:
        config_path: Path to the TOML config file
        model_class: Pydantic model class to validate the config (None for raw dict)
        raw: If True, return raw config dict without validation

    Returns:
        Raw dict by default, specified model_class if provided
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    from dlkit.infrastructure.config.core.sources import DLKitTomlSource

    source = DLKitTomlSource(config_path)
    config_data = source()

    # Return raw dict if explicitly requested
    if raw:
        return config_data

    # Return raw dict if no model class specified
    resolved_model_class: type[BaseModel] | None = model_class or _resolve_default_settings_class()
    if resolved_model_class is None:
        return config_data

    # Validate with the model class
    try:
        validated = resolved_model_class.model_validate(config_data)
    except Exception as e:
        raise ConfigValidationError(
            f"Failed to validate config with {resolved_model_class.__name__}: {e}",
            resolved_model_class.__name__,
            config_data,
        ) from e

    # Sync SESSION.root_dir to global environment for defensive fallback
    # This ensures SESSION.root_dir is respected even when PathOverrideContext is not active
    _sync_session_root_to_environment(validated)

    # No post-PathsResolver step; path resolution is environment/config-based
    return cast(T, validated)


def load_raw_config(config_path: Path | str) -> dict[str, Any]:
    """Load raw TOML config file as dict without validation.

    Args:
        config_path: Path to the TOML config file

    Returns:
        Raw config dictionary
    """
    return load_config(config_path, raw=True)


def load_sections_config(
    config_path: Path | str,
    section_configs: dict[str, type[BaseModel] | None] | list[str],
) -> dict[str, BaseModel]:
    """Load multiple sections from a TOML config file with eager validation.

    When ``section_configs`` is a mapping, the behaviour matches the previous
    implementation. A convenient shorthand now permits passing an iterable of
    section names, leveraging the predefined registry to resolve the
    corresponding ``BaseModel`` classes automatically.

    Args:
        config_path: Path to the TOML config file
        section_configs: Mapping of section names to their model classes *or*
            iterable of section names that use registered defaults.

    Returns:
        Dictionary mapping **uppercased** section names to model instances
        validated eagerly with defaults populated.

    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigSectionError: If any required section is missing or lacks a registered model
        ConfigValidationError: If validation=True and validation fails for any section

    Example:
        >>> configs = load_sections_config("config.toml", ["SESSION", "TRAINING"])
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    resolved_models = _resolve_section_models(section_configs)
    if not resolved_models:
        return {}

    from dlkit.infrastructure.config.core.sources import DLKitTomlSource

    section_names = list(resolved_models.keys())
    source = DLKitTomlSource(config_path, sections=section_names)
    sections_data = source()
    sections_data = {name.upper(): content for name, content in sections_data.items()}

    # Check for missing sections
    available_sections = get_available_sections(config_path)
    missing_sections = [name for name in section_names if name not in sections_data]

    if missing_sections:
        raise ConfigSectionError(
            f"Sections {missing_sections} not found in config file. "
            f"Available sections: {available_sections}",
            section_name=missing_sections[0] if missing_sections else None,
            available_sections=available_sections,
        )

    # Construct or validate each section based on validate parameter
    constructed_sections = {}
    for section_name, model_class in resolved_models.items():
        section_data = sections_data[section_name]
        try:
            constructed_sections[section_name] = model_class.model_validate(
                section_data, context={"strict": True}
            )
        except Exception as e:
            raise ConfigValidationError(
                f"Failed to validate section '{section_name}' with {model_class.__name__}: {e}",
                model_class.__name__,
                section_data,
            ) from e

    # If we loaded a full settings object (GeneralSettings), sync SESSION.root_dir
    # This handles partial loading where a complete GeneralSettings is returned
    if len(constructed_sections) > 1 and "SESSION" in constructed_sections:
        # Create a mock settings object with SESSION attribute for synchronization
        class _MockSettings:
            def __init__(self, session: Any) -> None:
                self.SESSION = session

        mock_settings = _MockSettings(constructed_sections["SESSION"])
        _sync_session_root_to_environment(mock_settings)

    return constructed_sections


@overload
def load_section_config[T: BaseModel](
    config_path: Path | str,
    model_class: type[T],
    section_name: str | None = None,
) -> T: ...


@overload
def load_section_config(
    config_path: Path | str,
    model_class: None = None,
    section_name: str | None = None,
) -> BaseModel: ...


def load_section_config[T: BaseModel](
    config_path: Path | str,
    model_class: type[T] | None = None,
    section_name: str | None = None,
) -> BaseModel | T:
    """Load a single config section with eager validation.

    Args:
        config_path: Path to the TOML config file
        model_class: Optional Pydantic model class to validate the section with
        section_name: Explicit section name (auto-detected from class name or
            registry when omitted)

    Returns:
        Model instance from the requested section
        validated eagerly

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ConfigSectionError: If the section is missing or lacks a registered model
        ConfigValidationError: If validation=True and validation fails for that section
        ValueError: If neither ``model_class`` nor ``section_name`` are provided
    """
    resolved_section = section_name
    resolved_model: type[BaseModel] | None = model_class

    if resolved_section is None and resolved_model is None:
        raise ValueError("Either model_class or section_name must be provided")

    if resolved_section is None and resolved_model is not None:
        resolved_section = get_section_name(resolved_model)

    if resolved_model is None and resolved_section is not None:
        resolved_model = get_model_class_for_section(resolved_section)

    if resolved_section is None:
        raise ValueError("Could not resolve section name from provided model_class")

    if resolved_model is None:
        raise ValueError(f"Could not find registered model for section: {resolved_section}")

    sections = load_sections_config(config_path, {resolved_section: resolved_model})
    return sections[resolved_section.upper()]


def check_section_exists(config_path: Path | str, section_name: str) -> bool:
    """Check if a section exists in the config file without full parsing.

    Args:
        config_path: Path to the TOML config file
        section_name: Name of section to check

    Returns:
        True if section exists, False otherwise

    Raises:
        FileNotFoundError: If config file doesn't exist

    Example:
        >>> if check_section_exists("config.toml", "PATHS"):
        ...     paths_config = load_section_config("config.toml", PathsSettings)
    """
    return section_name in get_available_sections(config_path)


def get_available_sections(config_path: Path | str) -> list[str]:
    """Get list of available sections without full config parsing.

    Args:
        config_path: Path to the TOML config file

    Returns:
        List of section names found in the config

    Raises:
        FileNotFoundError: If config file doesn't exist

    Example:
        >>> sections = get_available_sections("config.toml")
        >>> isinstance(sections, list)
        True
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return list(tomllib.load(f).keys())


def load_training_config_eager(config_path: Path | str) -> TrainingWorkflowConfig:
    """Load training config with eager Pydantic validation.

    This replaces the lazy loading pattern with eager validation while supporting
    programmatic section injection. Config sections present in TOML are validated
    immediately (fail-fast on typos/types). Missing optional sections can be
    injected programmatically before build time.

    Args:
        config_path: Path to TOML configuration file

    Returns:
        TrainingWorkflowConfig with eagerly validated sections

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config has type errors or invalid values
        ConfigValidationError: If required sections missing from TOML

    Example:
        ```python
        # Load partial config (eager validation of present fields)
        config = load_training_config_eager("config.toml")

        # Inject optional sections programmatically
        config = config.patch({"DATASET": DatasetSettings(features=(...))})

        # Validate completeness before build
        from dlkit.infrastructure.config.validators import validate_training_config_complete

        validate_training_config_complete(config)

        # Build components
        components = BuildFactory().build_components(config)
        ```
    """
    from dlkit.infrastructure.config.workflow_configs import TrainingWorkflowConfig

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Use the Dynaconf-backed loader to stay consistent with IO stack
    toml_data = load_config(config_path, raw=True)

    # Eager validation - fails fast on typos/types
    config = TrainingWorkflowConfig.model_validate(toml_data)

    # Sync root_dir to environment
    _sync_session_root_to_environment(config)

    return config


def load_inference_config_eager(config_path: Path | str) -> InferenceWorkflowConfig:
    """Load inference config with eager Pydantic validation.

    Args:
        config_path: Path to TOML configuration file

    Returns:
        InferenceWorkflowConfig with eagerly validated sections

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config has type errors or invalid values
    """
    from dlkit.infrastructure.config.workflow_configs import InferenceWorkflowConfig

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    toml_data = load_config(config_path, raw=True)
    config = InferenceWorkflowConfig.model_validate(toml_data)

    _sync_session_root_to_environment(config)

    return config


def load_optimization_config_eager(config_path: Path | str) -> OptimizationWorkflowConfig:
    """Load optimization config with eager Pydantic validation.

    Args:
        config_path: Path to TOML configuration file

    Returns:
        OptimizationWorkflowConfig with eagerly validated sections

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config has type errors or invalid values
    """
    from dlkit.infrastructure.config.workflow_configs import OptimizationWorkflowConfig

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    toml_data = load_config(config_path, raw=True)
    config = OptimizationWorkflowConfig.model_validate(toml_data)

    _sync_session_root_to_environment(config)

    return config
