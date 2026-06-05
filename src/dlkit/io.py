"""User-facing I/O namespace.

Thin re-exports from ``dlkit.infrastructure.io`` so users can write::

    from dlkit.io import load_array, locations

instead of::

    from dlkit.infrastructure.io import load_array
    from dlkit.infrastructure.io import locations
"""

from dlkit.infrastructure.io import (
    ConfigSectionError,
    ConfigValidationError,
    IWriter,
    JsonWriter,
    TomlWriter,
    WriterFactory,
    YamlWriter,
    ZarrLazyReader,
    check_section_exists,
    get_available_sections,
    load_array,
    load_config,
    load_raw_config,
    load_section_config,
    load_sections_config,
    load_split_indices,
    locations,
    read_table,
    register_section_mapping,
    reset_section_mappings,
    serialize_config_to_string,
    write_config,
)

__all__ = [
    # Zarr I/O
    "ZarrLazyReader",
    # Path locations
    "locations",
    # Array / table loading
    "load_array",
    "read_table",
    "load_split_indices",
    # Config I/O
    "load_config",
    "load_raw_config",
    "load_section_config",
    "load_sections_config",
    "write_config",
    "serialize_config_to_string",
    "check_section_exists",
    "get_available_sections",
    "register_section_mapping",
    "reset_section_mappings",
    "ConfigSectionError",
    "ConfigValidationError",
    # Writers
    "IWriter",
    "TomlWriter",
    "JsonWriter",
    "YamlWriter",
    "WriterFactory",
]
