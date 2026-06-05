"""Re-export shim — all public names are defined in the focused sub-modules.

Import from here for backward compatibility; prefer the specific modules for
new code:

    entry_protocols  — capability interfaces (IPathBased, IRuntimeGenerated, …)
    entry_base       — DataEntry ABC
    entry_types      — concrete types (Latent, ZarrEntry, ValueEntry, …)
    entry_factories  — type aliases and type guards
    data_roles       — DataRole enum
"""

from .data_roles import DataRole
from .entry_base import DataEntry
from .entry_factories import (
    AnyEntry,
    AnyPathEntry,
    is_feature,
    is_path_based,
    is_target,
    is_value_based,
)
from .entry_protocols import (
    IFeatureReference,
    IPathBased,
    IRuntimeGenerated,
    IValueBased,
)
from .entry_types import (
    AutoencoderTarget,
    CsvEntry,
    Hdf5Entry,
    Latent,
    NpyEntry,
    NpzEntry,
    ParquetEntry,
    PathBasedEntry,
    Prediction,
    ValueBasedEntry,
    ValueEntry,
    ZarrEntry,
)
from .transform_settings import TransformSettings

__all__ = [
    # Base
    "DataEntry",
    # Role enum
    "DataRole",
    # Protocols
    "IPathBased",
    "IValueBased",
    "IRuntimeGenerated",
    "IFeatureReference",
    # Abstract bases
    "PathBasedEntry",
    "ValueBasedEntry",
    # Format-specific path-based types
    "ZarrEntry",
    "NpyEntry",
    "NpzEntry",
    "CsvEntry",
    "ParquetEntry",
    "Hdf5Entry",
    # Unified value-based type
    "ValueEntry",
    # Special types
    "Latent",
    "AutoencoderTarget",
    "Prediction",
    # Type aliases
    "AnyPathEntry",
    "AnyEntry",
    # Type guards
    "is_path_based",
    "is_value_based",
    "is_feature",
    "is_target",
    # Re-exported for backward compat (was in the original module namespace)
    "TransformSettings",
]
