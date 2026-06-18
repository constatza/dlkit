"""Flexible dataset implementation for arbitrary feature/target configurations.

This module provides FlexibleDataset that loads an arbitrary set of feature
and target files based on data entry configurations.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, SupportsIndex, cast

from torch import Tensor

from dlkit.common.errors import (
    BatchComplianceError,  # noqa: F401  (re-exported for callers)
    PlaceholderNotResolvedError,  # noqa: F401  (re-exported for callers)
)
from dlkit.engine.data.sources import RoleSourceMap, build_role_source_map
from dlkit.infrastructure.config.entry_factories import AnyEntry

from .base import BaseDataset, register_dataset

if TYPE_CHECKING:
    from tensordict import TensorDict
    from tensordict.base import TensorDictBase


type _TensorMap = dict[str, Tensor]


def _build_nested_tensordict(
    feature_tensors: _TensorMap,
    target_tensors: _TensorMap,
    *,
    batch_size: list[int],
) -> TensorDict:
    """Build a nested TensorDict from typed feature and target tensor maps."""
    from tensordict import TensorDict

    return TensorDict(
        {
            "features": TensorDict(feature_tensors, batch_size=batch_size),  # type: ignore
            "targets": TensorDict(target_tensors, batch_size=batch_size),  # type: ignore
        },
        batch_size=batch_size,
    )


def _normalize_index(idx: int, n: int) -> int:
    """Normalize negative indices and validate bounds."""
    normalized = idx if idx >= 0 else n + idx
    if normalized < 0 or normalized >= n:
        raise IndexError(f"index {idx} out of range for dataset of size {n}")
    return normalized


def _normalize_indices(indices: list[int], n: int) -> list[int]:
    """Normalize and validate a batch of sample indices."""
    return [_normalize_index(int(idx), n) for idx in indices]


def _check_no_placeholders(entries: Sequence[AnyEntry]) -> None:
    """Raise PlaceholderNotResolvedError if any entry is a placeholder.

    Args:
        entries: Entries to check.

    Raises:
        PlaceholderNotResolvedError: If any entry has no path or value resolved.
    """
    for entry in entries:
        if hasattr(entry, "is_placeholder") and entry.is_placeholder():
            raise PlaceholderNotResolvedError(str(getattr(entry, "name", None) or "unknown"))


def _validate_no_scalars(entries: Sequence[AnyEntry]) -> None:
    """Validate that no entry produces a scalar (0-D) tensor.

    Args:
        entries: Entries to validate.

    Raises:
        BatchComplianceError: If any value-based entry holds a 0-D tensor/array.
    """
    from dlkit.infrastructure.config.entry_protocols import IValueBased

    for entry in entries:
        if not isinstance(entry, IValueBased):
            continue
        value = entry.get_value()
        if value is None:
            continue
        ndim = value.ndim if hasattr(value, "ndim") else 0
        if ndim == 0:
            raise BatchComplianceError(
                "Scalar (0-D) tensor entries are not allowed. "
                "Every entry must include the sample dimension N as its first axis. "
                "Reshape scalars to (N, 1) tensors before use."
            )


def _build_dataset_tensordict(role_map: RoleSourceMap, n: int) -> TensorDict:
    """Build a full TensorDict with batch_size=[n] from a RoleSourceMap.

    Loads all sources eagerly to produce a TensorDict that mirrors the legacy
    ``_dataset_td`` attribute expected by test consumers.

    Args:
        role_map: The role source map to load from.
        n: Number of samples.

    Returns:
        Nested TensorDict with ``batch_size=[n]``.
    """
    all_indices = list(range(n))
    feature_tensors: _TensorMap = {
        name: src.get_batch(all_indices) for name, src in role_map.features
    }
    target_tensors: _TensorMap = {
        name: src.get_batch(all_indices) for name, src in role_map.targets
    }
    return _build_nested_tensordict(feature_tensors, target_tensors, batch_size=[n])


@register_dataset
class FlexibleDataset(BaseDataset["TensorDict"]):
    """Dataset that loads an arbitrary set of feature and target entries.

    Entries are provided as DataEntry objects with ``data_role`` set to
    ``DataRole.FEATURE`` or ``DataRole.TARGET`` (NpyEntry, ZarrEntry,
    ValueEntry, etc.).

    The key used in __getitem__ output is the entry name, and the value is the
    tensor slice at the requested index.

    Precision handling is automatic via the global precision service. Use
    precision_override() context to control the dtype of loaded tensors.

    Supported File Formats:
    - NumPy arrays: .npy (single array), .npz (multi-array)
    - PyTorch tensors: .pt, .pth
    - Text files: .txt, .csv
    - Zarr arrays: native zarr v3 stores (lazy, indexed)

    For .npz files with multiple arrays, the entry name is used as the array key
    to select which array to load from the file.

    Supports:
    - Path-based entries: Data loaded from files
    - Value-based entries: In-memory data
    - Placeholder entries: Must be resolved before use (raises PlaceholderNotResolvedError)

    Single Responsibility: Load and manage dataset lifecycle (NO validation).
    Validation is handled by entry constructors.

    Examples:
        >>> from dlkit.infrastructure.config.entry_types import ValueEntry
        >>> from dlkit.infrastructure.config.data_roles import DataRole
        >>> feat = ValueEntry(name="x", value=x_tensor, data_role=DataRole.FEATURE)
        >>> targ = ValueEntry(name="y", value=y_tensor, data_role=DataRole.TARGET)
        >>> dataset = FlexibleDataset(entries=[feat, targ])
    """

    def __init__(
        self,
        entries: Sequence[AnyEntry],
    ) -> None:
        """Initialize FlexibleDataset with a list of entries.

        Args:
            entries: Unified list of DataEntry objects with ``data_role`` set.
                Role filtering is performed via ``is_feature()`` / ``is_target()``.

        Raises:
            BatchComplianceError: If a value-based entry is scalar (0-D), or if
                sources report conflicting sample counts.  File-backed scalar
                arrays are not detected at construction time — they raise on
                first access.
            ValueError: If no entries are provided.
            PlaceholderNotResolvedError: If placeholder entry without value.
            TypeError: If entry is value-based but ``get_value()`` returns None.
        """
        if not entries:
            raise ValueError("At least one feature or target entry is required")

        _check_no_placeholders(entries)
        _validate_no_scalars(entries)

        self._role_map: RoleSourceMap = build_role_source_map(entries)
        self._length: int = self._role_map.n_samples
        self._feature_names: tuple[str, ...] = tuple(n for n, _ in self._role_map.features)
        self._target_names: tuple[str, ...] = tuple(n for n, _ in self._role_map.targets)

    def __len__(self) -> int:
        """Return number of samples in dataset.

        Returns:
            Number of samples (first dimension of tensors)
        """
        return self._length

    def __getitem__(self, idx: SupportsIndex) -> TensorDict:  # ty: ignore[invalid-method-override]
        """Get sample at index.

        Args:
            idx: Sample index

        Returns:
            TensorDict with feature and target nested TensorDicts (batch_size=[])
        """
        i = _normalize_index(int(idx), self._length)
        feature_tensors = {name: src.get_item(i) for name, src in self._role_map.features}
        target_tensors = {name: src.get_item(i) for name, src in self._role_map.targets}
        return _build_nested_tensordict(feature_tensors, target_tensors, batch_size=[])

    def __getitems__(self, indices: list[int]) -> TensorDict:
        """Get a batched TensorDict for a list of indices.

        This path is used by PyTorch DataLoader for map-style datasets when
        auto-collation is enabled, allowing source reads to be batched once
        per batch rather than one sample at a time.

        Args:
            indices: List of sample indices.

        Returns:
            Batched TensorDict with ``batch_size=[len(indices)]``.

        Raises:
            ValueError: If ``indices`` is empty.
        """
        if not indices:
            raise ValueError("indices must be non-empty")
        idxs = _normalize_indices(indices, self._length)
        feature_tensors = {name: src.get_batch(idxs) for name, src in self._role_map.features}
        target_tensors = {name: src.get_batch(idxs) for name, src in self._role_map.targets}
        return _build_nested_tensordict(feature_tensors, target_tensors, batch_size=[len(idxs)])

    @functools.cached_property
    def _dataset_td(self) -> TensorDict:
        """Full dataset as a TensorDict, built lazily on first access.

        Returns:
            Nested TensorDict with ``batch_size=[n]`` containing all features
            and targets.
        """
        return _build_dataset_tensordict(self._role_map, self._length)

    @property
    def collate_fn(self) -> Callable[[list[TensorDictBase]], TensorDict]:
        """Collate function for DataLoaders.

        Returns:
            ``collate_tensordict`` function, which merges a list of TensorDict
            samples (or a pre-batched TensorDict from ``__getitems__``) into a
            single batched TensorDict.
        """
        return collate_tensordict


def collate_tensordict(batch: list[TensorDictBase] | TensorDictBase) -> TensorDict:
    """Collate samples into a batched TensorDict.

    Used as the collate_fn for DataLoaders with FlexibleDataset.

    Args:
        batch: Either a list of TensorDict samples from ``__getitem__`` or a
            pre-batched TensorDict returned by ``__getitems__``.

    Returns:
        Batched TensorDict.

    Raises:
        BatchComplianceError: If stacked batch size does not match expected count
            when list collation is used.
    """
    from tensordict import TensorDictBase
    from tensordict import stack as td_stack

    if isinstance(batch, TensorDictBase):
        return cast("TensorDict", batch)

    result = td_stack(batch, dim=0)
    if result.batch_size[0] != len(batch):
        raise BatchComplianceError(
            f"Collated batch size {result.batch_size[0]} does not match expected {len(batch)}."
        )
    return cast("TensorDict", result)
