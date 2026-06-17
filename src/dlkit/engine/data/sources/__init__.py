"""Source implementations for the DLKit data layer.

This subpackage provides the concrete ``ArraySource`` implementations used
by dataset and dataloader code to access feature and target arrays uniformly,
regardless of whether the backing store is an in-memory tensor, a NumPy file,
or a lazy zarr array.

Public symbols
--------------
BroadcastSource
    Wraps a single-sample source to broadcast the same tensor across any
    requested batch size.
EagerFileSource
    Loads an entire array from disk at construction time and serves slices
    from an in-memory tensor.
TensorSource
    Wraps an already-resolved ``torch.Tensor`` as an ``ArraySource``.
NamedSources
    Type alias for an immutable ordered sequence of ``(name, ArraySource)`` pairs.
RoleSourceMap
    Frozen dataclass holding feature and target ``NamedSources`` with a
    consistent ``n_samples`` property.
source_from_entry
    Dispatch a single ``AnyEntry`` to its ``ArraySource`` implementation.
build_role_source_map
    Build a validated ``RoleSourceMap`` from a sequence of entries.
"""

from .base import BroadcastSource, NamedSources
from .eager import EagerFileSource
from .role_map import BatchComplianceError, RoleSourceMap, build_role_source_map, source_from_entry
from .tensor import TensorSource

__all__ = [
    "BatchComplianceError",
    "BroadcastSource",
    "EagerFileSource",
    "NamedSources",
    "RoleSourceMap",
    "TensorSource",
    "build_role_source_map",
    "source_from_entry",
]
