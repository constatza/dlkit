from dlkit.infrastructure.zarr import ILazyReader, ZarrLazyReader

from . import locations  # centralized path policy
from .arrays import load_array
from .config import (
    ConfigValidationError,
    check_section_exists,
    get_available_sections,
    load_config,
    load_raw_config,
    serialize_config_to_string,
    write_config,
)
from .index import load_split_indices
from .path_resolver import PathResolver
from .tables import read_table
from .tensor_entries import (
    TensorDataEntry,
    convert_totensor_entries,
    to_tensor_entry,
)
from .writers import IWriter, JsonWriter, TomlWriter, WriterFactory, YamlWriter

__all__ = [
    # Data loading
    "load_array",
    "read_table",
    "load_split_indices",
    # Tensor entries
    "TensorDataEntry",
    "to_tensor_entry",
    "convert_totensor_entries",
    # Low-level config utilities
    "load_config",
    "load_raw_config",
    "check_section_exists",
    "get_available_sections",
    "write_config",
    "serialize_config_to_string",
    "ConfigValidationError",
    # Path resolution
    "PathResolver",
    "locations",
    # Zarr I/O
    "ILazyReader",
    "ZarrLazyReader",
    # Writers
    "IWriter",
    "TomlWriter",
    "JsonWriter",
    "YamlWriter",
    "WriterFactory",
]
