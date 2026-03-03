# High-level config loading API (re-exported from tools.config)
from dlkit.tools.config.factories import load_sections, load_settings

from . import locations  # centralized path policy
from .arrays import load_array
from .config import (
    ConfigSectionError,
    ConfigValidationError,
    check_section_exists,
    get_available_sections,
    load_config,
    load_raw_config,
    load_section_config,  # low-level, not in __all__
    load_sections_config,  # low-level, not in __all__
    register_section_mapping,
    reset_section_mappings,
    write_config,
)
from .index import load_split_indices
from dlkit.tools.config.data_entries import SparseFeature

from .sparse import (
    PackFiles,
    PackManifest,
    SparseFormat,
    is_sparse_pack_dir,
    open_sparse_pack,
    register_manifest_schema,
    save_sparse_pack,
    validate_sparse_pack,
)
from .tables import read_table

__all__ = [
    # High-level config loading API
    "load_settings",
    "load_sections",
    # Data loading
    "load_array",
    "read_table",
    "load_split_indices",
    # Low-level config utilities
    "load_config",
    "load_raw_config",
    "check_section_exists",
    "get_available_sections",
    "reset_section_mappings",
    "write_config",
    "register_section_mapping",
    "ConfigSectionError",
    "ConfigValidationError",
    # Path locations
    "locations",
    # Sparse pack I/O
    "open_sparse_pack",
    "is_sparse_pack_dir",
    "save_sparse_pack",
    "validate_sparse_pack",
    "PackFiles",
    "PackManifest",
    "register_manifest_schema",
    "SparseFormat",
    "SparseFeature",
]
