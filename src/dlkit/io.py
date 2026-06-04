"""User-facing I/O namespace.

Thin re-exports from ``dlkit.infrastructure.io`` so users can write::

    from dlkit.io import open_array_pack, save_array_pack, locations

instead of::

    from dlkit.infrastructure.io import open_array_pack, save_array_pack
    from dlkit.infrastructure.io import locations
"""

from dlkit.infrastructure.io import (
    ArrayPackFormat,
    ConfigSectionError,
    ConfigValidationError,
    IWriter,
    JsonWriter,
    TomlWriter,
    WriterFactory,
    YamlWriter,
    check_section_exists,
    get_available_sections,
    load_array,
    load_config,
    load_raw_config,
    load_section_config,
    load_sections_config,
    load_split_indices,
    locations,
    open_array_pack,
    read_table,
    register_format,
    register_section_mapping,
    reset_section_mappings,
    save_array_pack,
    serialize_config_to_string,
    write_array_pack,
    write_config,
)

__all__ = [
    # Array pack I/O (packs API)
    "open_array_pack",
    "write_array_pack",
    "save_array_pack",
    "ArrayPackFormat",
    "register_format",
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
