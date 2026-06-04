"""Re-export shim — all public names are defined in the focused sub-modules.

Import from here for backward compatibility; prefer the specific modules for
new code:

    entry_protocols  — capability interfaces (IPathBased, IWritable, …)
    entry_base       — DataEntry ABC
    entry_types      — concrete types (PathFeature, ValueTarget, Latent, …)
    entry_factories  — Feature(), Target(), ContextFeature(), type guards
"""

from .entry_base import DataEntry
from .entry_factories import (
    ContextFeature,
    Feature,
    FeatureType,
    Matrix,
    Sparse,
    Target,
    TargetType,
    has_feature_reference,
    is_feature_entry,
    is_path_based,
    is_runtime_generated,
    is_target_entry,
    is_value_based,
    is_writable,
)
from .entry_protocols import (
    IFeatureReference,
    IPathBased,
    IRuntimeGenerated,
    IValueBased,
    IWritable,
)
from .entry_types import (
    AutoencoderTarget,
    Latent,
    MatrixFeature,
    PathBasedEntry,
    PathFeature,
    PathTarget,
    Prediction,
    SparseFeature,
    ValueBasedEntry,
    ValueFeature,
    ValueTarget,
)
from .transform_settings import TransformSettings

__all__ = [
    # Base
    "DataEntry",
    # Protocols
    "IPathBased",
    "IValueBased",
    "IWritable",
    "IRuntimeGenerated",
    "IFeatureReference",
    # Abstract bases
    "PathBasedEntry",
    "ValueBasedEntry",
    # Path-based types
    "PathFeature",
    "MatrixFeature",
    "SparseFeature",
    "PathTarget",
    # Value-based types
    "ValueFeature",
    "ValueTarget",
    # Special types
    "Latent",
    "AutoencoderTarget",
    "Prediction",
    # Factories
    "Feature",
    "FeatureType",
    "Matrix",
    "Sparse",  # deprecated alias for Matrix
    "Target",
    "TargetType",
    "ContextFeature",
    # Type guards
    "is_feature_entry",
    "is_target_entry",
    "is_path_based",
    "is_value_based",
    "is_writable",
    "is_runtime_generated",
    "has_feature_reference",
    # Re-exported for backward compat (was in the original module namespace)
    "TransformSettings",
]
