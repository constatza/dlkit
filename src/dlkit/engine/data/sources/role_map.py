"""Role-partitioned source map and entry-to-source dispatch.

Provides ``RoleSourceMap``, an immutable frozen dataclass keying ordered
``NamedSources`` pairs by data role (features / targets), and two factory
functions:

* ``source_from_entry`` — dispatches a single ``AnyEntry`` to its
  ``ArraySource`` implementation.
* ``build_role_source_map`` — constructs a validated ``RoleSourceMap``
  from a sequence of entries, automatically wrapping singleton sources
  in ``BroadcastSource``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from dlkit.common.sources import ArraySource
from dlkit.infrastructure.config.entry_factories import AnyEntry, is_feature, is_target
from dlkit.infrastructure.config.entry_protocols import IValueBased
from dlkit.infrastructure.precision.service import PrecisionService

from ._helpers import _require_name
from .base import BroadcastSource, NamedSources
from .eager import EagerFileSource
from .tensor import TensorSource


class BatchComplianceError(ValueError):
    """Raised when sources violate batch-shape invariants."""


@dataclass(frozen=True)
class RoleSourceMap:
    """Immutable ordered map of sources keyed by data role.

    Uses tuples of ``(name, ArraySource)`` pairs rather than dicts to
    guarantee immutability.  The ``n_samples`` property excludes
    ``BroadcastSource`` instances so that constant-sized singleton
    sources (e.g. a fixed bias vector) do not interfere with canonical-N
    resolution.

    Attributes:
        features: Ordered ``(name, ArraySource)`` pairs for feature entries.
        targets: Ordered ``(name, ArraySource)`` pairs for target entries.

    Example:
        >>> import torch
        >>> from dlkit.engine.data.sources.tensor import TensorSource
        >>> src = TensorSource(torch.zeros(100, 4))
        >>> rsm = RoleSourceMap(features=(("x", src),), targets=())
        >>> rsm.n_samples
        100
    """

    features: NamedSources
    targets: NamedSources

    def features_dict(self) -> dict[str, ArraySource]:
        """Return features as a mutable ``dict`` for ad-hoc lookups.

        Returns:
            ``dict`` mapping feature name to its ``ArraySource``.
        """
        return dict(self.features)

    def targets_dict(self) -> dict[str, ArraySource]:
        """Return targets as a mutable ``dict`` for ad-hoc lookups.

        Returns:
            ``dict`` mapping target name to its ``ArraySource``.
        """
        return dict(self.targets)

    @property
    def n_samples(self) -> int:
        """Canonical number of samples, derived from non-broadcast sources.

        Skips any ``BroadcastSource`` entries (which always report
        ``n_samples == 1``) and validates that all remaining sources agree.

        Returns:
            Consistent sample count across all non-broadcast sources.

        Raises:
            ValueError: If there are no non-broadcast sources, or if
                multi-sample sources report conflicting sample counts.
        """
        multi = [
            src.n_samples
            for _, src in (*self.features, *self.targets)
            if not isinstance(src, BroadcastSource)
        ]
        if not multi:
            raise ValueError("RoleSourceMap has no non-broadcast sources; cannot resolve n_samples")
        if len(set(multi)) > 1:
            raise BatchComplianceError(
                f"All entries must share the same first dimension N, "
                f"but found conflicting sizes: {set(multi)}"
            )
        return multi[0]


def source_from_entry(entry: AnyEntry) -> ArraySource:
    """Dispatch an entry to its ``ArraySource``.  Exactly three cases.

    Cases:
        1. ``IValueBased`` — wraps ``entry.get_value()`` in a
           ``TensorSource`` after precision-casting.
        2. ``open_reader()`` returns ``ArraySource`` — use directly
           (e.g. ``ZarrLazyReader``).
        3. ``open_reader()`` returns ``Path`` — construct an
           ``EagerFileSource`` that loads the whole array at call time.

    Args:
        entry: A ``DataEntry`` with role and data-access interface.

    Returns:
        An ``ArraySource`` for this entry's data.

    Raises:
        TypeError: If ``entry`` is ``IValueBased`` but ``get_value()``
            returns ``None`` (placeholder mode).
    """
    if isinstance(entry, IValueBased):
        value = entry.get_value()
        if value is None:
            raise TypeError(f"ValueBased entry '{entry.name}' returned None from get_value()")
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        tensor = PrecisionService().cast_tensor(tensor)
        return TensorSource(tensor)

    # PathBasedEntry branch: open_reader() returns ArraySource | Path
    reader = entry.open_reader()
    match reader:
        case Path():
            # Only pass array_key for multi-array formats (e.g. .npz).
            # For single-array formats (.npy, .csv, etc.) the loader does
            # not accept array_key and PathBasedEntry.array_key would
            # return the entry name as a fallback — not a format key.
            is_multi_array = reader.suffix.lower() == ".npz"
            array_key = getattr(entry, "array_key", None) if is_multi_array else None
            return EagerFileSource(
                reader,
                dtype=getattr(entry, "dtype", None),
                array_key=array_key,
                **getattr(entry, "load_kwargs", {}),
            )
        case _:
            # Already an ArraySource (e.g. ZarrLazyReader)
            return reader


def _maybe_broadcast(name: str, src: ArraySource) -> tuple[str, ArraySource]:
    """Wrap ``src`` in ``BroadcastSource`` when it is a singleton.

    Args:
        name: The entry name.
        src: The resolved ``ArraySource``.

    Returns:
        ``(name, src)`` unchanged when ``src.n_samples != 1``, otherwise
        ``(name, BroadcastSource(src))``.
    """
    if src.n_samples == 1:
        return name, BroadcastSource(src)
    return name, src


def build_role_source_map(entries: Sequence[AnyEntry]) -> RoleSourceMap:
    """Build a ``RoleSourceMap`` from a sequence of entries.

    Steps:
        1. Partition entries by ``is_feature`` / ``is_target``.
        2. Call ``source_from_entry`` for each entry.
        3. Resolve the canonical sample count from multi-sample sources.
        4. Wrap singleton sources in ``BroadcastSource``.
        5. Freeze the result into an immutable ``RoleSourceMap``.

    Args:
        entries: Sequence of ``DataEntry`` objects.

    Returns:
        A frozen ``RoleSourceMap``.

    Raises:
        ValueError: If multi-sample sources have conflicting ``n_samples``.
    """
    feature_entries = [e for e in entries if is_feature(e)]
    target_entries = [e for e in entries if is_target(e)]

    raw_features: list[tuple[str, ArraySource]] = [
        (_require_name(e), source_from_entry(e)) for e in feature_entries
    ]
    raw_targets: list[tuple[str, ArraySource]] = [
        (_require_name(e), source_from_entry(e)) for e in target_entries
    ]

    all_sources = [src for _, src in (*raw_features, *raw_targets)]
    multi_n = {src.n_samples for src in all_sources if src.n_samples != 1}

    if len(multi_n) > 1:
        raise BatchComplianceError(
            f"All entries must share the same first dimension N, "
            f"but found conflicting sizes: {multi_n}"
        )

    features = tuple(_maybe_broadcast(n, s) for n, s in raw_features)
    targets = tuple(_maybe_broadcast(n, s) for n, s in raw_targets)

    return RoleSourceMap(features=features, targets=targets)


__all__ = [
    "BatchComplianceError",
    "RoleSourceMap",
    "build_role_source_map",
    "source_from_entry",
]
