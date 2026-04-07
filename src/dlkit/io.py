"""User-facing I/O namespace.

Thin re-exports from ``dlkit.infrastructure.io`` so users can write::

    from dlkit.io import open_sparse_pack, save_sparse_pack, locations

instead of::

    from dlkit.infrastructure.io import open_sparse_pack, save_sparse_pack
    from dlkit.infrastructure.io import locations
"""

from dlkit.infrastructure.io import (
    ConfigSectionError,
    ConfigValidationError,
    IWriter,
    JsonWriter,
    PackFiles,
    PackManifest,
    SparseFormat,
    TomlWriter,
    WriterFactory,
    YamlWriter,
    check_section_exists,
    get_available_sections,
    is_sparse_pack_dir,
    load_array,
    load_config,
    load_raw_config,
    load_section_config,
    load_sections_config,
    load_split_indices,
    locations,
    open_sparse_pack,
    read_table,
    register_manifest_schema,
    register_section_mapping,
    reset_section_mappings,
    save_sparse_pack,
    serialize_config_to_string,
    validate_sparse_pack,
    write_config,
)

__all__ = [
    # Sparse pack I/O
    "open_sparse_pack",
    "save_sparse_pack",
    "validate_sparse_pack",
    "is_sparse_pack_dir",
    "PackFiles",
    "PackManifest",
    "SparseFormat",
    "register_manifest_schema",
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
