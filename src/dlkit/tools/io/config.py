"""TOML config loader with subsettings support using dynaconf."""

from collections.abc import Mapping, Sequence
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any, TypeVar, cast, overload, get_origin, get_args, Union
import sys
import torch

from dynaconf import Dynaconf
from pydantic import BaseModel
from enum import Enum
from tomlkit import document, table, dumps

from .parsers import PartialTOMLParser

# Python 3.10+ UnionType compatibility
try:
    from types import UnionType
except ImportError:
    UnionType = type(None)


def _sync_session_root_to_environment(settings: Any) -> None:
    """Synchronize SESSION.root_dir to global DLKitEnvironment if appropriate.

    This provides a defensive fallback when PathOverrideContext is not active.
    Respects precedence: DLKIT_ROOT_DIR env var > SESSION.root_dir > CWD.

    Args:
        settings: Loaded settings object (GeneralSettings or similar)
    """
    try:
        import os
        from dlkit.tools.config.environment import env as global_environment
        from dlkit.tools.utils.system_utils import coerce_root_dir_to_absolute
        from loguru import logger

        # Only update if DLKitEnvironment doesn't already have root_dir from env var
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
            "Synchronized SESSION.root_dir to DLKitEnvironment for fallback path resolution",
            session_root_dir=str(normalized_root),
        )
    except Exception as e:
        # Non-fatal - path resolution will fall back to CWD if this fails
        from loguru import logger
        logger.debug(f"Failed to sync SESSION.root_dir to environment (non-fatal): {e}")


# Type variable for BaseModel subclasses used throughout the module
T = TypeVar("T", bound=BaseModel)


class ConfigSectionError(ValueError):
    """Raised when a config section is missing or invalid."""

    def __init__(
        self,
        message: str,
        section_name: str | None = None,
        available_sections: list[str] | None = None,
    ):
        super().__init__(message)
        self.section_name = section_name
        self.available_sections = available_sections or []


class ConfigValidationError(ValueError):
    """Raised when config validation fails."""

    def __init__(self, message: str, model_class: str, section_data: dict[str, Any] | None = None):
        super().__init__(message)
        self.model_class = model_class
        self.section_data = section_data or {}


# Section mapping registry: Maps between Pydantic model classes and TOML section names
T = TypeVar("T", bound=BaseModel)


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


def _resolve_default_settings_class() -> type[BaseModel] | None:
    """Lazily import GeneralSettings without top-level coupling."""

    module_name = "dlkit.tools.config.general_settings"
    module = sys.modules.get(module_name)

    if module is None:
        if find_spec(module_name) is None:
            return None
        module = import_module(module_name)

    general = getattr(module, "GeneralSettings", None)
    if isinstance(general, type) and issubclass(general, BaseModel):
        return general
    return None


def _initialize_default_mappings() -> None:
    """Initialize default section mappings for common settings classes."""

    try:
        from dlkit.tools.config.session_settings import SessionSettings
        from dlkit.tools.config.components.model_components import ModelComponentSettings
        from dlkit.tools.config.training_settings import TrainingSettings
        from dlkit.tools.config.datamodule_settings import DataModuleSettings
        from dlkit.tools.config.dataset_settings import DatasetSettings
        from dlkit.tools.config.optuna_settings import OptunaSettings
        from dlkit.tools.config.paths_settings import PathsSettings
        from dlkit.tools.config.extras_settings import ExtrasSettings

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


# Initialize default mappings
_initialize_default_mappings()


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

    settings = Dynaconf(
        settings_files=[str(config_path)],
        load_dotenv=False,
    )

    config_data = settings.to_dict()

    # Flatten any redundant nesting created by dotted assignments like
    # "MODEL.checkpoint = ..." which Dynaconf may represent as
    # {"MODEL": {"name": ..., "MODEL": {"checkpoint": ...}}}
    # Merge inner dict into the parent to match expected Pydantic structure.
    try:
        for section in (
            "SESSION",
            "MODEL",
            "DATASET",
            "DATAMODULE",
            "TRAINING",
            "MLFLOW",
            "OPTUNA",
            "ENVIRONMENT",
            "EXTRAS",
        ):
            sec_val = config_data.get(section)
            if (
                isinstance(sec_val, dict)
                and section in sec_val
                and isinstance(sec_val[section], dict)
            ):
                inner = sec_val.pop(section)
                # Don't overwrite existing keys unless explicitly set in inner
                for k, v in inner.items():
                    sec_val[k] = v
    except Exception:
        # Best-effort flattening; ignore if structure is unexpected
        pass

    # No PATHS injection; environment root is authoritative

    # Filter out Dynaconf metadata that shouldn't be passed to Pydantic
    dynaconf_metadata_keys = {"LOAD_DOTENV", "ENV_FOR_DYNACONF", "ROOT_PATH_FOR_DYNACONF"}
    config_data = {k: v for k, v in config_data.items() if k not in dynaconf_metadata_keys}

    # Minimal pre-resolution for fields that use FilePath/DirectoryPath validators.
    try:
        from pathlib import Path as _Path
        from dlkit.core.datatypes.urls import tilde_expand_strict
        from dlkit.interfaces.api.overrides.path_context import get_current_path_context

        config_dir = Path(config_path).resolve().parent

        def _compute_root_dict(cfg: dict[str, Any]) -> _Path:
            """Compute effective root with standard precedence.

            Priority:
            1) DLKIT_ROOT_DIR env var (via DLKitEnvironment)
            2) ENVIRONMENT.root_dir in config (resolved relative to config file dir if relative)
            3) SESSION.root_dir in config (legacy-friendly; resolved relative to config file dir)
            4) Parent directory of the config file
            5) Current working directory
            """
            # 1) Process environment (explicit env wins)
            # 0) CLI/path override context (highest priority)
            try:
                ctx = get_current_path_context()
                if ctx and getattr(ctx, "root_dir", None):
                    return _Path(ctx.root_dir).resolve()
            except Exception:
                pass

            # 1) SESSION.root_dir (formal config override)
            ses = cfg.get("SESSION")
            if isinstance(ses, dict) and ses.get("root_dir"):
                p = _Path(tilde_expand_strict(str(ses["root_dir"])))
                return (p if p.is_absolute() else (config_dir / p)).resolve()

            # 2) Config directory, then 3) CWD
            return config_dir if config_dir.exists() else _Path.cwd().resolve()

        def _is_url(value: Any) -> bool:
            return isinstance(value, str) and "://" in value

        # No directory creation during config load

        def _process_path_field(value: Any, root: _Path) -> Any:
            if not isinstance(value, str):
                return value
            if _is_url(value):
                # Leave URL values as-is; Pydantic types will validate later
                return value
            expanded = tilde_expand_strict(value)
            path = _Path(expanded)
            if not path.is_absolute():
                path = (root / path).resolve()
            return str(path)

        root = _compute_root_dict(config_data)

        # 1) TRAINING.trainer.default_root_dir (absolute path only)
        trainer = (
            (config_data.get("TRAINING") or {}).get("trainer")
            if isinstance(config_data.get("TRAINING"), dict)
            else None
        )
        if isinstance(trainer, dict) and "default_root_dir" in trainer:
            val = trainer["default_root_dir"]
            processed = _process_path_field(val, root)
            trainer["default_root_dir"] = processed

        # 2) MODEL.checkpoint
        model_sec = config_data.get("MODEL")
        if isinstance(model_sec, dict) and "checkpoint" in model_sec:
            model_sec["checkpoint"] = _process_path_field(model_sec["checkpoint"], root)

        # 3) DATASET.split.filepath and DATASET.features/targets paths relative to DATASET.root
        dataset_sec = config_data.get("DATASET")
        if isinstance(dataset_sec, dict):
            # Compute dataset base directory if provided
            dataset_root_val = dataset_sec.get("root_dir", dataset_sec.get("root"))
            dataset_base: _Path | None = None
            if isinstance(dataset_root_val, str) and dataset_root_val:
                processed_root = _process_path_field(dataset_root_val, root)
                dataset_base = _Path(processed_root)

            split = dataset_sec.get("split")
            if isinstance(split, dict) and "filepath" in split:
                fp_val = split["filepath"]
                if isinstance(fp_val, str) and dataset_base and not _is_url(fp_val):
                    p = _Path(tilde_expand_strict(fp_val))
                    if not p.is_absolute():
                        split["filepath"] = str((dataset_base / p).resolve())
                    else:
                        split["filepath"] = str(p)
                else:
                    split["filepath"] = _process_path_field(fp_val, root)

            # 4) DATASET.features[*].path and DATASET.targets[*].path
            for list_key in ("features", "targets"):
                entries = dataset_sec.get(list_key)
                if isinstance(entries, (list, tuple)):
                    new_entries = []
                    changed = False
                    for item in entries:
                        if not isinstance(item, dict):
                            new_entries.append(item)
                            continue
                        if "path" in item:
                            path_val = item["path"]
                            if isinstance(path_val, str) and dataset_base and not _is_url(path_val):
                                p = _Path(tilde_expand_strict(path_val))
                                if not p.is_absolute():
                                    new_path = str((dataset_base / p).resolve())
                                else:
                                    new_path = str(p)
                            else:
                                new_path = _process_path_field(path_val, root)
                            if new_path is not item["path"]:
                                item = dict(item)
                                item["path"] = new_path
                                changed = True
                        new_entries.append(item)
                    if changed:
                        dataset_sec[list_key] = new_entries

    except Exception:
        pass

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


def _to_toml_compatible(value: Any) -> Any:
    """Convert values into TOML-serializable primitives.

    - Converts Path to str
    - Converts Enum to its value
    - Converts torch.dtype to str
    - Recursively processes dicts and sequences
    """
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    # Handle Pydantic AnyUrl objects
    if hasattr(value, "__class__") and "AnyUrl" in str(type(value)):
        return str(value)
    # Handle pydantic_core Url objects
    try:
        from pydantic_core import Url as _CoreUrl  # type: ignore

        if isinstance(value, _CoreUrl):  # type: ignore[arg-type]
            return str(value)
    except Exception:
        pass
    # Handle torch.dtype objects by converting to string
    if isinstance(value, torch.dtype):
        return str(value)
    # Fallback for torch objects that might not be dtype but still need string conversion
    if hasattr(value, "__module__") and value.__module__ and "torch" in value.__module__:
        return str(value)
    if isinstance(value, dict):
        return {k: _to_toml_compatible(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_to_toml_compatible(v) for v in value if v is not None]
    return value


def write_config(
    config: BaseModel | dict[str, Any],
    output_path: Path | str,
    *,
    by_alias: bool = True,
    exclude_none: bool = True,
    exclude_unset: bool = False,
    sort_sections: bool = True,
) -> Path:
    """Write a DLKit configuration to a TOML file.

    Accepts a Pydantic model (e.g., GeneralSettings) or a raw dict mapping
    top-level section names (e.g., SESSION, MODEL, PATHS) to their contents.

    Args:
        config: Pydantic settings model or raw dict to write
        output_path: Destination TOML file path
        by_alias: Dump using field aliases (e.g., PATHS.root instead of root_dir)
        exclude_none: Exclude fields that are None
        exclude_unset: Exclude fields that were not explicitly set (for Pydantic models only)
        sort_sections: Write sections in sorted order for stable diffs

    Returns:
        Path to the written TOML file
    """
    output_path = Path(output_path)

    if isinstance(config, BaseModel):
        data = config.model_dump(
            by_alias=by_alias, exclude_none=exclude_none, exclude_unset=exclude_unset
        )
    else:
        data = dict(config)

    # Create TOML document and add sections
    doc = document()
    items = sorted(data.items(), key=lambda kv: kv[0]) if sort_sections else data.items()
    for section, content in items:
        if content is None:
            continue
        sec_table = table()
        for k, v in _to_toml_compatible(content).items():
            sec_table.add(k, v)
        doc.add(section, sec_table)

    output_path.write_text(dumps(doc))
    return output_path


def _preprocess_sections(config_path: Path, sections: dict[str, Any]) -> dict[str, Any]:
    """Apply path resolution and normalization to section data."""

    if not sections:
        return sections

    try:
        from pathlib import Path as _Path
        from dlkit.core.datatypes.urls import tilde_expand_strict
        from dlkit.interfaces.api.overrides.path_context import get_current_path_context
    except Exception:
        # If imports fail (e.g., minimal environment), skip preprocessing gracefully
        return sections

    config_dir = config_path.resolve().parent

    def _is_url(value: Any) -> bool:
        return isinstance(value, str) and "://" in value

    def _process_path_field(value: Any, base: _Path) -> Any:
        if not isinstance(value, str):
            return value
        if _is_url(value):
            return value
        expanded = tilde_expand_strict(value)
        path_obj = _Path(expanded)
        if not path_obj.is_absolute():
            path_obj = (base / path_obj).resolve()
        return str(path_obj)

    def _resolve_root_dir(session_data: dict[str, Any] | None) -> _Path:
        try:
            ctx = get_current_path_context()
            if ctx and getattr(ctx, "root_dir", None):
                return _Path(ctx.root_dir).resolve()
        except Exception:
            pass

        if isinstance(session_data, dict):
            candidate = session_data.get("root_dir")
            if isinstance(candidate, str) and candidate:
                base = _process_path_field(candidate, config_dir)
                return _Path(base).resolve()

        return config_dir if config_dir.exists() else _Path.cwd().resolve()

    processed_sections = dict(sections)
    session_data = processed_sections.get("SESSION")
    if isinstance(session_data, dict) and "root_dir" in session_data:
        session_copy = dict(session_data)
        session_copy["root_dir"] = _process_path_field(session_copy["root_dir"], config_dir)
        processed_sections["SESSION"] = session_copy
        session_data = session_copy

    root_dir = _resolve_root_dir(session_data)

    def _preprocess_training_section(data: dict[str, Any]) -> dict[str, Any]:
        trainer = data.get("trainer")
        if isinstance(trainer, dict) and "default_root_dir" in trainer:
            trainer_copy = dict(trainer)
            trainer_copy["default_root_dir"] = _process_path_field(
                trainer_copy["default_root_dir"], root_dir
            )
            data = dict(data)
            data["trainer"] = trainer_copy
        return data

    def _preprocess_model_section(data: dict[str, Any]) -> dict[str, Any]:
        if "checkpoint" in data:
            model_copy = dict(data)
            model_copy["checkpoint"] = _process_path_field(model_copy["checkpoint"], root_dir)
            return model_copy
        return data

    def _preprocess_dataset_section(data: dict[str, Any]) -> dict[str, Any]:
        dataset_copy = dict(data)
        dataset_root_val = dataset_copy.get("root_dir", dataset_copy.get("root"))
        dataset_base: _Path | None = None
        if isinstance(dataset_root_val, str) and dataset_root_val:
            processed_root = _process_path_field(dataset_root_val, root_dir)
            dataset_copy["root_dir"] = processed_root
            dataset_base = _Path(processed_root)

        split = dataset_copy.get("split")
        if isinstance(split, dict) and "filepath" in split:
            split_copy = dict(split)
            value = split_copy["filepath"]
            if isinstance(value, str) and dataset_base and not _is_url(value):
                p = _Path(tilde_expand_strict(value))
                if not p.is_absolute():
                    split_copy["filepath"] = str((dataset_base / p).resolve())
                else:
                    split_copy["filepath"] = str(p)
            else:
                split_copy["filepath"] = _process_path_field(value, root_dir)
            dataset_copy["split"] = split_copy

        for list_key in ("features", "targets"):
            entries = dataset_copy.get(list_key)
            if isinstance(entries, (list, tuple)):
                new_entries = []
                changed = False
                for item in entries:
                    if not isinstance(item, dict):
                        new_entries.append(item)
                        continue
                    if "path" in item:
                        new_item = dict(item)
                        path_val = new_item["path"]
                        if isinstance(path_val, str) and dataset_base and not _is_url(path_val):
                            p_val = _Path(tilde_expand_strict(path_val))
                            if not p_val.is_absolute():
                                new_item["path"] = str((dataset_base / p_val).resolve())
                            else:
                                new_item["path"] = str(p_val)
                        else:
                            new_item["path"] = _process_path_field(path_val, root_dir)
                        if new_item["path"] != item.get("path"):
                            changed = True
                        new_entries.append(new_item)
                    else:
                        new_entries.append(item)
                if changed:
                    dataset_copy[list_key] = new_entries

        return dataset_copy

    def _preprocess_paths_section(data: dict[str, Any]) -> dict[str, Any]:
        paths_copy = dict(data)
        for key, value in list(paths_copy.items()):
            if isinstance(value, str) and value:
                paths_copy[key] = _process_path_field(value, root_dir)
        return paths_copy

    if isinstance(processed_sections.get("TRAINING"), dict):
        processed_sections["TRAINING"] = _preprocess_training_section(
            processed_sections["TRAINING"]
        )

    if isinstance(processed_sections.get("MODEL"), dict):
        processed_sections["MODEL"] = _preprocess_model_section(processed_sections["MODEL"])

    if isinstance(processed_sections.get("DATASET"), dict):
        processed_sections["DATASET"] = _preprocess_dataset_section(processed_sections["DATASET"])

    if isinstance(processed_sections.get("PATHS"), dict):
        processed_sections["PATHS"] = _preprocess_paths_section(processed_sections["PATHS"])

    return processed_sections


def _resolve_section_models(
    section_configs: Mapping[str, type[BaseModel] | None] | Sequence[str],
) -> dict[str, type[BaseModel]]:
    """Resolve mapping of section names to model classes using registry defaults."""

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
        resolved[section_name] = resolved_model
    return resolved


def _construct_without_defaults(
    model_class: type[BaseModel],
    data: dict[str, Any]
) -> BaseModel:
    """Construct Pydantic model with type coercion but no default filling.

    This function enables lazy loading of partial configs by:
    - Recursively handling nested Pydantic models for proper type construction
    - Applying type coercion (str→Path, str→Enum, dict→NestedModel, etc.)
    - NOT filling in defaults for missing fields
    - Only including fields explicitly present in data

    Args:
        model_class: Pydantic model class to construct
        data: Raw dictionary data from TOML file

    Returns:
        Model instance with only fields from data (no defaults)

    Example:
        >>> # TOML has: [DATASET]
        >>> # (empty section)
        >>> result = _construct_without_defaults(DatasetSettings, {})
        >>> # result has NO split field with default ratios
        >>> # vs model_validate would add split=IndexSplitSettings(test_ratio=0.15, ...)
    """
    # Get field info to identify nested Pydantic models
    processed_data = {}
    fields_set = set()

    for key, value in data.items():
        field_info = model_class.model_fields.get(key)
        if field_info is None:
            # Extra field (like in ExtrasSettings with extra="allow") - pass through
            processed_data[key] = value
            fields_set.add(key)
            continue

        # Mark field as explicitly set
        fields_set.add(key)

        # Check if field is a nested Pydantic model
        field_type = field_info.annotation

        # Handle Optional[SomeModel], Union[SomeModel, None], etc.
        origin = get_origin(field_type)
        if origin in (Union, UnionType):
            # Extract non-None type from Union
            args = get_args(field_type)
            field_type = next((arg for arg in args if arg is not type(None)), field_type)

        # If nested dict and field expects BaseModel, recursively construct
        if isinstance(value, dict) and isinstance(field_type, type) and issubclass(field_type, BaseModel):
            # Recursively construct nested model without defaults
            processed_data[key] = _construct_without_defaults(field_type, value)
        elif isinstance(value, (list, tuple)):
            # Handle lists/tuples of nested models
            processed_items = []
            for item in value:
                # Check if the field is a container of BaseModel instances
                # e.g., tuple[Feature, ...] or list[Target]
                item_type = field_type
                item_origin = get_origin(field_type)
                if item_origin in (tuple, list):
                    item_args = get_args(field_type)
                    if item_args:
                        # Get first arg type (e.g., Feature from tuple[Feature, ...])
                        item_type = item_args[0]

                if isinstance(item, dict) and isinstance(item_type, type) and issubclass(item_type, BaseModel):
                    processed_items.append(_construct_without_defaults(item_type, item))
                else:
                    processed_items.append(item)

            # Preserve original type (tuple vs list) AND respect field type annotation
            item_origin = get_origin(field_type)
            if item_origin == tuple or isinstance(value, tuple):
                # Field expects tuple or original value was tuple
                processed_data[key] = tuple(processed_items)
            else:
                # Field expects list or no specific requirement
                processed_data[key] = processed_items
        else:
            # For primitives and simple types, pass through as-is
            # Pydantic's model_construct will handle basic type coercion
            processed_data[key] = value

    # Use model_construct with _fields_set to track explicitly set fields
    # CRITICAL: _fields_set tells Pydantic which fields were explicitly set
    instance = model_class.model_construct(_fields_set=fields_set, **processed_data)

    # WORKAROUND: model_construct still fills defaults for fields with default_factory
    # We need to remove those fields if they weren't explicitly in the data
    # This preserves true partial loading without unwanted defaults from factories
    for field_name, field_info in model_class.model_fields.items():
        if field_name not in fields_set:
            # Field was not in the original data
            # Only remove fields with default_factory (like IndexSplitSettings())
            # Keep simple defaults (like () or "string") - they're lightweight and expected
            if field_info.default_factory is not None:
                # Remove the factory-created default that model_construct added
                # Use object.__delattr__ to bypass Pydantic's __delattr__
                try:
                    object.__delattr__(instance, field_name)
                except AttributeError:
                    # Field might not exist, that's fine
                    pass

    return instance


def load_sections_config(
    config_path: Path | str,
    section_configs: Mapping[str, type[BaseModel] | None] | Sequence[str],
    *,
    validate: bool = False,
) -> dict[str, BaseModel]:
    """Load multiple sections from a TOML config file with optional validation.

    Supports lazy loading (default) where Pydantic validation is deferred,
    or strict loading where validation happens immediately.

    When ``section_configs`` is a mapping, the behaviour matches the previous
    implementation. A convenient shorthand now permits passing an iterable of
    section names, leveraging the predefined registry to resolve the
    corresponding ``BaseModel`` classes automatically.

    Args:
        config_path: Path to the TOML config file
        section_configs: Mapping of section names to their model classes *or*
            iterable of section names that use registered defaults.
        validate: If True, use model_validate() for strict validation (eager).
            If False (default), use model_construct() for lazy loading (deferred).

    Returns:
        Dictionary mapping **uppercased** section names to model instances
        (validated if validate=True, unvalidated if validate=False)

    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigSectionError: If any required section is missing or lacks a registered model
        ConfigValidationError: If validation=True and validation fails for any section

    Example:
        >>> # Lazy loading (default) - no validation yet
        >>> configs = load_sections_config("config.toml", ["SESSION", "TRAINING"])
        >>> # Strict loading - validates immediately
        >>> configs = load_sections_config("config.toml", ["SESSION"], validate=True)
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    resolved_models = _resolve_section_models(section_configs)
    if not resolved_models:
        return {}

    # Use partial parser for efficiency
    parser = PartialTOMLParser()
    section_names = list(resolved_models.keys())

    # Get raw section data
    sections_data = parser.parse_sections(config_path, section_names)
    sections_data = {name.upper(): content for name, content in sections_data.items()}

    # Check for missing sections
    available_sections = parser.get_available_sections(config_path)
    missing_sections = [name for name in section_names if name not in sections_data]

    if missing_sections:
        raise ConfigSectionError(
            f"Sections {missing_sections} not found in config file. "
            f"Available sections: {available_sections}",
            section_name=missing_sections[0] if missing_sections else None,
            available_sections=available_sections,
        )

    sections_data = _preprocess_sections(config_path, sections_data)

    # Construct or validate each section based on validate parameter
    constructed_sections = {}
    for section_name, model_class in resolved_models.items():
        section_data = sections_data[section_name]
        try:
            if validate:
                # Strict mode: full validation with defaults filled
                constructed_sections[section_name] = model_class.model_validate(section_data)
            else:
                # Lazy mode: type coercion WITHOUT default filling
                # Uses smart constructor that handles nested models and type coercion
                # but ONLY includes fields explicitly present in TOML
                constructed_sections[section_name] = _construct_without_defaults(
                    model_class, section_data
                )
        except Exception as e:
            mode = "validate" if validate else "construct"
            raise ConfigValidationError(
                f"Failed to {mode} section '{section_name}' with {model_class.__name__}: {e}",
                model_class.__name__,
                section_data,
            ) from e

    # If we loaded a full settings object (GeneralSettings), sync SESSION.root_dir
    # This handles partial loading with validate=True where a complete GeneralSettings is returned
    if len(constructed_sections) > 1 and "SESSION" in constructed_sections:
        # Create a mock settings object with SESSION attribute for synchronization
        class _MockSettings:
            pass
        mock_settings = _MockSettings()
        mock_settings.SESSION = constructed_sections["SESSION"]
        _sync_session_root_to_environment(mock_settings)

    return constructed_sections


@overload
def load_section_config(
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


def load_section_config(
    config_path: Path | str,
    model_class: type[T] | None = None,
    section_name: str | None = None,
    *,
    validate: bool = False,
) -> BaseModel | T:
    """Load a single config section with optional validation.

    Args:
        config_path: Path to the TOML config file
        model_class: Optional Pydantic model class to validate the section with
        section_name: Explicit section name (auto-detected from class name or
            registry when omitted)
        validate: If True, use model_validate() for strict validation (eager).
            If False (default), use model_construct() for lazy loading (deferred).

    Returns:
        Model instance from the requested section
        (validated if validate=True, unvalidated if validate=False)

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

    sections = load_sections_config(config_path, {resolved_section: resolved_model}, validate=validate)
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
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    parser = PartialTOMLParser()
    available_sections = parser.get_available_sections(config_path)
    return section_name in available_sections


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
        >>> print(f"Available sections: {sections}")
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    parser = PartialTOMLParser()
    return parser.get_available_sections(config_path)
