"""TOML config loader public re-export surface."""

from dlkit.tools.io._config_serializer import serialize_config_to_string, write_config
from dlkit.tools.io.config_errors import ConfigSectionError, ConfigValidationError
from dlkit.tools.io.config_loader import (
    check_section_exists,
    get_available_sections,
    load_config,
    load_inference_config_eager,
    load_optimization_config_eager,
    load_raw_config,
    load_section_config,
    load_sections_config,
    load_training_config_eager,
)
from dlkit.tools.io.config_section_registry import (
    get_model_class_for_section,
    get_section_name,
    register_section_mapping,
    reset_section_mappings,
)

__all__ = [
    "ConfigSectionError",
    "ConfigValidationError",
    "check_section_exists",
    "get_available_sections",
    "get_model_class_for_section",
    "get_section_name",
    "load_config",
    "load_inference_config_eager",
    "load_optimization_config_eager",
    "load_raw_config",
    "load_section_config",
    "load_sections_config",
    "load_training_config_eager",
    "register_section_mapping",
    "reset_section_mappings",
    "serialize_config_to_string",
    "write_config",
]
