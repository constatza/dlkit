"""Flexible dataset implementation for arbitrary feature/target configurations.

This module provides FlexibleDataset that loads an arbitrary set of feature
and target files based on data entry configurations.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, SupportsIndex, cast

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from dlkit.common.errors import PlaceholderNotResolvedError  # noqa: F401  (re-exported for callers)
from dlkit.infrastructure.config.entry_factories import AnyEntry, is_feature, is_target
from dlkit.infrastructure.config.normalized_entry import NormalizedEntry as _NormalizedEntry
from dlkit.infrastructure.io import load_array
from dlkit.infrastructure.io.arrays import _numpy_array_to_tensor
from dlkit.infrastructure.zarr import ILazyReader

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
            "features": TensorDict(cast(Any, feature_tensors), batch_size=batch_size),
            "targets": TensorDict(cast(Any, target_tensors), batch_size=batch_size),
        },
        batch_size=batch_size,
    )


class BatchComplianceError(ValueError):
    """Raised when dataset entries violate batch-shape invariants.

    Enforces the contract that every entry must carry an explicit sample
    dimension N as its first axis, and all entries must agree on N.
    """


def _normalize_entries(
    entries: Any,
) -> dict[str, _NormalizedEntry]:
    """Extract normalized data sources from DataEntry objects.

    Each entry's normalize() method produces a _NormalizedEntry containing
    the appropriate source (ILazyReader, Path, Tensor, or ndarray).

    Args:
        entries: Collection of DataEntry objects

    Returns:
        Dictionary mapping entry name to _NormalizedEntry.

    Raises:
        TypeError: If entries is a raw dict
        PlaceholderNotResolvedError: If any entry is a placeholder
    """
    result: dict[str, _NormalizedEntry] = {}
    if entries is None:
        return result

    if isinstance(entries, dict):
        raise TypeError(
            "FlexibleDataset no longer accepts raw dicts. "
            "Use NpyEntry, ZarrEntry, or ValueEntry instead."
        )

    for item in entries:
        if isinstance(item, dict):
            raise TypeError(
                "FlexibleDataset no longer accepts raw dicts. "
                "Use NpyEntry, ZarrEntry, or ValueEntry instead."
            )
        if not hasattr(item, "name") or item.name is None:
            raise TypeError(
                f"Entry {type(item).__name__} has no name. "
                "All entries must have a name set before normalization."
            )
        result[item.name] = item.normalize()

    return result


def _load_or_convert_tensor(
    source: Path | Tensor | np.ndarray,
    dtype: torch.dtype | None = None,
    array_key: str | None = None,
    **load_kwargs: Any,
) -> Tensor:
    """Pure function: convert source to torch.Tensor with dtype handling.

    Handles both file paths (production) and in-memory data (testing).
    Respects precision context via PrecisionService for dtype resolution.

    Args:
        source: File path OR in-memory tensor/array
        dtype: Target dtype (uses PrecisionService if None)
        array_key: For .npz files, the array name to extract
        **load_kwargs: Extra kwargs forwarded to load_array() (e.g. mmap_mode)

    Returns:
        torch.Tensor with appropriate dtype
    """
    # Case 1: Already a torch.Tensor or numpy array (in-memory data)
    if isinstance(source, (torch.Tensor, np.ndarray)):
        resolved_dtype = dtype
        if resolved_dtype is None:
            from dlkit.infrastructure.precision.service import get_precision_service

            precision_service = get_precision_service()
            resolved_dtype = precision_service.get_torch_dtype()
        if isinstance(source, np.ndarray):
            return _numpy_array_to_tensor(source, dtype=resolved_dtype)
        return source.to(dtype=resolved_dtype)

    # Case 2: File path - delegate to existing load_array()
    # load_array() already handles PrecisionService integration
    # For .npz files, pass array_key to select specific array
    if isinstance(source, Path) and source.suffix.lower() == ".npz":
        return load_array(source, dtype=dtype, array_key=array_key, **load_kwargs)
    return load_array(source, dtype=dtype, **load_kwargs)


def _get_source_dtype() -> torch.dtype:
    """Get the active precision dtype from PrecisionService.

    Returns:
        torch.dtype: Dtype from the active precision context.
    """
    from dlkit.infrastructure.precision.service import get_precision_service

    return get_precision_service().get_torch_dtype()


def _compute_source_fingerprint(
    all_maps: dict[str, _NormalizedEntry],
    dtype: torch.dtype,
) -> str:
    """Compute SHA-256 fingerprint over source files and dtype.

    Args:
        all_maps: Mapping from entry name to _NormalizedEntry (Path sources only).
        dtype: Target tensor dtype included in the fingerprint.

    Returns:
        str: Full hex-digest fingerprint string.
    """
    h = hashlib.sha256()
    for name in sorted(all_maps):
        ne = all_maps[name]
        assert isinstance(ne.source, Path)
        h.update(name.encode())
        h.update(ne.source.as_posix().encode())
        h.update(str(ne.source.stat().st_mtime_ns).encode())
    h.update(str(dtype).encode())
    return h.hexdigest()


def _read_cache_fingerprint(cache_dir: Path) -> str | None:
    """Read the stored fingerprint from a cache directory.

    Args:
        cache_dir: Cache directory path.

    Returns:
        str | None: Fingerprint string, or None if not present.
    """
    fp_file = cache_dir / "dlkit_fingerprint.txt"
    return fp_file.read_text().strip() if fp_file.exists() else None


def _write_cache_fingerprint(cache_dir: Path, fp: str) -> None:
    """Persist fingerprint into a cache directory.

    Args:
        cache_dir: Cache directory path.
        fp: Fingerprint string to write.
    """
    (cache_dir / "dlkit_fingerprint.txt").write_text(fp)


def _write_entry_to_memmap(
    name: str,
    source: Path,
    array_key: str | None,
    group_dir: Path,
    dtype: torch.dtype,
    chunk_size: int,
) -> Any:
    """OOM-safe write of a single array entry into a MemoryMappedTensor file.

    For .npy sources the source is read via numpy mmap (no full RAM load).
    For all other sources the tensor is loaded normally then chunked into the file.

    Args:
        name: Entry name (used for output filename).
        source: Source file path.
        array_key: Array key for .npz files; equals entry name by convention.  ``None`` for non-npz.
        group_dir: Directory to write the .memmap file into.
        dtype: Target tensor dtype.
        chunk_size: Number of rows written per iteration.

    Returns:
        MemoryMappedTensor: Disk-backed tensor filled with the entry data.
    """
    from tensordict.memmap import MemoryMappedTensor

    suffix = source.suffix.lower()
    src: np.ndarray
    if suffix == ".npy":
        src = np.load(str(source), mmap_mode="r")
    else:
        tensor = _load_or_convert_tensor(source, dtype=dtype, array_key=array_key)
        src = tensor.numpy()

    shape = src.shape
    filename = str(group_dir / f"{name}.memmap")
    mmt = MemoryMappedTensor.empty(list(shape), dtype=dtype, filename=filename)
    for start in range(0, shape[0], chunk_size):
        end = min(start + chunk_size, shape[0])
        mmt[start:end] = torch.as_tensor(np.array(src[start:end])).to(dtype=dtype)
    return mmt


def _determine_n_total(
    dense_tensors: dict[str, Any],
    pack_bindings: dict[str, ILazyReader] | None = None,
) -> int:
    """Pure function: resolve the canonical sample count N from available tensors.

    Precedence: dense tensors (features + targets) → zarr lazy readers (n_samples > 1).

    Args:
        dense_tensors: Already-written tensors keyed by entry name.
        pack_bindings: ILazyReader instances.  Readers with
            ``n_samples > 1`` contribute to N resolution when no dense entries
            are present.

    Returns:
        int: Canonical dataset size N.

    Raises:
        BatchComplianceError: If dense sizes disagree or pack sample counts disagree.
        ValueError: If N cannot be inferred (no dense entries, all-broadcast packs).
    """
    dense_n_set = {int(t.shape[0]) for t in dense_tensors.values()}
    if dense_n_set:
        if len(dense_n_set) > 1:
            raise BatchComplianceError(
                f"All entries must share the same first dimension N, "
                f"but found differing sizes: {sorted(dense_n_set)}."
            )
        return next(iter(dense_n_set))

    resolved_pack_bindings = pack_bindings or {}
    pack_ns = {r.n_samples for r in resolved_pack_bindings.values() if r.n_samples > 1}
    if pack_ns:
        if len(pack_ns) > 1:
            raise BatchComplianceError(
                f"Matrix pack entries have differing sample counts: {sorted(pack_ns)}."
            )
        return next(iter(pack_ns))

    raise ValueError("No entries provided to determine dataset size.")


def _validate_memmap_entries(
    all_maps: dict[str, _NormalizedEntry],
) -> None:
    """Raise ValueError if any entry is not a file path.

    Callers must strip ``ZarrLazyReader`` entries before calling this function;
    zarr readers bypass the memmap cache entirely.

    Args:
        all_maps: Mapping from entry name to _NormalizedEntry.
            Must not contain ``ZarrLazyReader`` entries (strip them before calling).

    Raises:
        ValueError: If any source is not a Path.
    """
    for name, ne in all_maps.items():
        if not isinstance(ne.source, Path):
            raise ValueError(
                f"Entry '{name}' is not file-backed. "
                "memmap_cache_dir requires PathBasedEntry entries."
            )


def _partition_entry_map(
    entry_map: dict[str, _NormalizedEntry],
) -> tuple[dict[str, Tensor], dict[str, ILazyReader]]:
    """Split a normalised entry map into eager tensors and lazy zarr readers.

    Args:
        entry_map: Mapping from entry name to _NormalizedEntry.

    Returns:
        Tuple of (dense_tensors, lazy_readers) where dense_tensors contains
        pre-loaded tensors and lazy_readers contains ILazyReader instances.
    """
    dense: dict[str, Tensor] = {}
    lazy: dict[str, ILazyReader] = {}
    for name, ne in entry_map.items():
        if isinstance(ne.source, ILazyReader):
            lazy[name] = ne.source
        else:
            dense[name] = _load_or_convert_tensor(
                ne.source, array_key=ne.array_key, **ne.load_kwargs
            )
    return dense, lazy


def _make_empty_dataset_tensordict(n: int) -> TensorDict:
    """Create an empty dataset TensorDict with batch size [n]."""
    from tensordict import TensorDict

    return TensorDict(
        {
            "features": TensorDict({}, batch_size=[n]),
            "targets": TensorDict({}, batch_size=[n]),
        },
        batch_size=[n],
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


def _inject_lazy_readers(
    td: TensorDict,
    *,
    idx: int | list[int],
    readers: dict[str, ILazyReader],
    namespace: str,
) -> TensorDict:
    """Inject lazy-reader tensors into a TensorDict at the given nested namespace.

    Args:
        td: TensorDict to enrich (mutated in place).
        idx: Single sample index (int) or batch of indices (list[int]).
        readers: Lazy zarr readers keyed by entry name.
        namespace: Nested key prefix (e.g. ``"features"`` or ``"targets"``).

    Returns:
        The enriched TensorDict.
    """
    if not readers:
        return td

    for name, reader in readers.items():
        td[namespace, name] = reader[idx]
    return td


def _assemble_dataset_tensordict(
    feat_tensors: dict[str, Tensor],
    targ_tensors: dict[str, Tensor],
    feat_names: tuple[str, ...],
    targ_names: tuple[str, ...],
) -> tuple[TensorDict, int]:
    """Validate batch compliance once and assemble nested TensorDict.

    Performs the single authoritative compliance check: no scalars, all lengths
    agree, N >= 1.  Feature and target sub-dicts are ordered by feat_names /
    targ_names to match the original entry ordering.

    Args:
        feat_tensors: Feature tensor map.
        targ_tensors: Target tensor map.
        feat_names: Feature entry names in insertion order.
        targ_names: Target entry names in insertion order.

    Returns:
        Tuple of (assembled TensorDict with batch_size=[N], canonical N).

    Raises:
        BatchComplianceError: If any tensor is scalar, sizes disagree, or N < 1.
        ValueError: If no tensors are provided.
    """
    all_tensors = {**feat_tensors, **targ_tensors}
    for tensor in all_tensors.values():
        if tensor.dim() == 0:
            raise BatchComplianceError(
                "Scalar (0-D) tensor entries are not allowed. "
                "Every entry must include the sample dimension N as its first axis. "
                "Reshape scalars to (N, 1) tensors before use."
            )

    sizes = {int(t.size(0)) for t in all_tensors.values()}
    if len(sizes) > 1:
        raise BatchComplianceError(
            f"All entries must share the same first dimension N, "
            f"but found differing sizes: {sorted(sizes)}."
        )
    if not sizes:
        raise ValueError("At least one feature or target entry is required after validation")

    n = next(iter(sizes))
    if n < 1:
        raise BatchComplianceError("Entries must contain at least one sample (N >= 1).")

    feature_batch: _TensorMap = {name: feat_tensors[name] for name in feat_names}
    target_batch: _TensorMap = {name: targ_tensors[name] for name in targ_names}
    td = _build_nested_tensordict(feature_batch, target_batch, batch_size=[n])
    return td, n


def _load_entry_from_memmap(filename: Path, info: dict[str, Any]) -> Tensor:
    """Load a single entry from a .memmap file using numpy mmap (zero-copy).

    Args:
        filename: Path to the .memmap file.
        info: Metadata dict with ``shape`` (list[int]) and ``dtype`` (str).

    Returns:
        Tensor: Memory-mapped tensor backed by the file.
    """
    shape = tuple(info["shape"])
    torch_dtype: torch.dtype = getattr(torch, info["dtype"].replace("torch.", ""))
    np_dtype = torch.empty(0, dtype=torch_dtype).numpy().dtype
    # mode='c' (copy-on-write): writable view so torch.from_numpy won't warn;
    # pages are still loaded lazily from disk on read — OOM benefit is preserved.
    arr = np.memmap(str(filename), dtype=np_dtype, mode="c", shape=shape)
    # torch.from_numpy keeps a reference to arr, preventing GC / mmap closure
    return torch.from_numpy(arr)


def _build_memmap_cache(
    feat_map: dict[str, _NormalizedEntry],
    targ_map: dict[str, _NormalizedEntry],
    cache_dir: Path,
    dtype: torch.dtype,
    chunk_size: int = 5_000,
) -> TensorDict:
    """Build an OOM-safe memmap cache for a dataset.

    Args:
        feat_map: Feature entry map: name → _NormalizedEntry (Path sources only).
        targ_map: Target entry map: name → _NormalizedEntry (Path sources only).
        cache_dir: Root directory for the cache.
        dtype: Target tensor dtype for all entries.
        chunk_size: Rows per write iteration.

    Returns:
        TensorDict: Dataset TensorDict backed by the new memmap files.

    Raises:
        BatchComplianceError: If feature/target sample counts disagree.
        ValueError: If N cannot be inferred from available entries.
    """
    features_dir = cache_dir / "features"
    targets_dir = cache_dir / "targets"
    for d in (cache_dir, features_dir, targets_dir):
        d.mkdir(parents=True, exist_ok=True)

    meta: dict[str, dict[str, Any]] = {"features": {}, "targets": {}}
    feat_tensors: _TensorMap = {}
    targ_tensors: _TensorMap = {}

    for name, ne in feat_map.items():
        assert isinstance(ne.source, Path)
        mmt = _write_entry_to_memmap(name, ne.source, ne.array_key, features_dir, dtype, chunk_size)
        feat_tensors[name] = mmt
        meta["features"][name] = {"shape": list(mmt.shape), "dtype": str(dtype)}

    for name, ne in targ_map.items():
        assert isinstance(ne.source, Path)
        mmt = _write_entry_to_memmap(name, ne.source, ne.array_key, targets_dir, dtype, chunk_size)
        targ_tensors[name] = mmt
        meta["targets"][name] = {"shape": list(mmt.shape), "dtype": str(dtype)}

    n = _determine_n_total({**feat_tensors, **targ_tensors})

    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    return _build_nested_tensordict(feat_tensors, targ_tensors, batch_size=[n])


def _load_memmap_from_cache(cache_dir: Path) -> TensorDict:
    """Reconstruct a TensorDict from an existing memmap cache directory.

    Args:
        cache_dir: Cache directory previously built by ``_build_memmap_cache``.

    Returns:
        TensorDict: Dataset TensorDict backed by memory-mapped files.
    """
    with open(cache_dir / "meta.json") as f:
        meta = json.load(f)

    def _load_group(group_key: str, group_dir: Path) -> tuple[dict[str, Tensor], int]:
        tensors: dict[str, Tensor] = {}
        n = 0
        for name, info in meta[group_key].items():
            n = info["shape"][0]
            tensors[name] = _load_entry_from_memmap(group_dir / f"{name}.memmap", info)
        return tensors, n

    feat_tensors, n = _load_group("features", cache_dir / "features")
    targ_tensors, _ = _load_group("targets", cache_dir / "targets")
    return _build_nested_tensordict(feat_tensors, targ_tensors, batch_size=[n])


def _load_or_build_memmap(
    feat_map: dict[str, _NormalizedEntry],
    targ_map: dict[str, _NormalizedEntry],
    cache_dir: Path,
    dtype: torch.dtype,
    chunk_size: int = 5_000,
) -> TensorDict:
    """Load a memmap dataset from cache, rebuilding if stale or absent.

    Cache invalidation is fingerprint-based: any change to source file mtimes
    or dtype triggers a full wipe-and-rebuild (no stale sub-directories).

    Args:
        feat_map: Feature entry map: name → _NormalizedEntry (Path sources only).
        targ_map: Target entry map: name → _NormalizedEntry (Path sources only).
        cache_dir: Directory for memmap cache files.
        dtype: Target tensor dtype (included in fingerprint).
        chunk_size: Rows written per iteration when building the cache.

    Returns:
        TensorDict: Dataset TensorDict backed by memory-mapped files.
    """
    all_maps: dict[str, _NormalizedEntry] = {**feat_map, **targ_map}
    fingerprint = _compute_source_fingerprint(all_maps, dtype)
    stored_fp = _read_cache_fingerprint(cache_dir)

    if stored_fp == fingerprint and (cache_dir / "meta.json").exists():
        logger.info(f"Loading memmap dataset from cache: {cache_dir}")
        return _load_memmap_from_cache(cache_dir)

    if cache_dir.exists() and stored_fp != fingerprint:
        logger.info(f"Cache fingerprint mismatch — rebuilding memmap cache at {cache_dir}")
        shutil.rmtree(cache_dir)
    else:
        logger.info(f"Building memmap cache at {cache_dir}")

    td = _build_memmap_cache(feat_map, targ_map, cache_dir, dtype, chunk_size)
    _write_cache_fingerprint(cache_dir, fingerprint)
    return td


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
        *,
        memmap_cache_dir: Path | None = None,
        memmap_chunk_size: int = 5_000,
    ) -> None:
        """Initialize FlexibleDataset with a list of entries.

        Args:
            entries: Unified list of DataEntry objects with ``data_role`` set.
                Role filtering is performed via ``is_feature()`` / ``is_target()``.
            memmap_cache_dir: If set, load dataset via OS memory-mapped files stored in
                this directory.  Entries must be file-backed (PathBasedEntry).
                Zarr entries bypass the memmap cache — zarr handles OOM natively.
                The cache is invalidated when source files or dtype change.
            memmap_chunk_size: Rows written per iteration when building the memmap cache.
                Bounds peak RAM to ``chunk_size × feature_width × sizeof(dtype)``.
                Ignored when ``memmap_cache_dir`` is ``None``.

        Raises:
            BatchComplianceError: If any entry is scalar (0-D) or N sizes do not agree.
            ValueError: If no features or targets are provided, or a non-file-backed
                entry is used with memmap_cache_dir.
            PlaceholderNotResolvedError: If placeholder entry without value.
            TypeError: If raw dicts are passed (use entry constructors instead).
        """
        feature_entries: list[Any] = [e for e in entries if is_feature(e)]
        target_entries: list[Any] = [e for e in entries if is_target(e)]

        feat_map = _normalize_entries(feature_entries)
        targ_map = _normalize_entries(target_entries)

        if not feat_map and not targ_map:
            raise ValueError("At least one feature or target entry is required")

        self._feature_names: tuple[str, ...] = tuple(feat_map.keys())
        self._target_names: tuple[str, ...] = tuple(targ_map.keys())
        self._feature_lazy_readers: dict[str, ILazyReader] = {}
        self._target_lazy_readers: dict[str, ILazyReader] = {}

        if memmap_cache_dir is not None:
            # ILazyReader entries bypass the memmap cache — zarr handles OOM natively.
            # Strip zarr readers from both maps before passing to memmap validation.
            feat_zarr = {
                n: ne.source for n, ne in feat_map.items() if isinstance(ne.source, ILazyReader)
            }
            targ_zarr = {
                n: ne.source for n, ne in targ_map.items() if isinstance(ne.source, ILazyReader)
            }
            self._feature_lazy_readers = feat_zarr
            self._target_lazy_readers = targ_zarr
            non_zarr_feat = {k: v for k, v in feat_map.items() if k not in feat_zarr}
            non_zarr_targ = {k: v for k, v in targ_map.items() if k not in targ_zarr}
            _validate_memmap_entries({**non_zarr_feat, **non_zarr_targ})
            active_dtype = _get_source_dtype()
            self._dataset_td = _load_or_build_memmap(
                non_zarr_feat,
                non_zarr_targ,
                Path(memmap_cache_dir),
                active_dtype,
                memmap_chunk_size,
            )
        else:
            dense_feats, feat_lazy = _partition_entry_map(feat_map)
            dense_targs, targ_lazy = _partition_entry_map(targ_map)
            self._feature_lazy_readers = feat_lazy
            self._target_lazy_readers = targ_lazy

            # Resolve canonical N: dense entries → zarr readers.
            non_scalar_dense = {
                k: v for k, v in {**dense_feats, **dense_targs}.items() if v.dim() > 0
            }
            n = _determine_n_total(non_scalar_dense, {**feat_lazy, **targ_lazy})

            if dense_feats or dense_targs:
                self._dataset_td, _ = _assemble_dataset_tensordict(
                    dense_feats,
                    dense_targs,
                    tuple(dense_feats.keys()),
                    tuple(dense_targs.keys()),
                )
            else:
                # Pack-only dataset: keep an empty dense base, inject per sample.
                self._dataset_td = _make_empty_dataset_tensordict(n)

        self._length = int(self._dataset_td.batch_size[0])

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
        sample_index = _normalize_index(int(idx), self._length)
        sample = cast("TensorDict", self._dataset_td[sample_index])
        _inject_lazy_readers(
            sample, idx=sample_index, readers=self._feature_lazy_readers, namespace="features"
        )
        _inject_lazy_readers(
            sample, idx=sample_index, readers=self._target_lazy_readers, namespace="targets"
        )
        return sample

    def __getitems__(self, indices: list[int]) -> TensorDict:
        """Get a batched TensorDict for a list of indices.

        This path is used by PyTorch DataLoader for map-style datasets when
        auto-collation is enabled, allowing pack feature injection once per batch.
        """
        if not indices:
            raise ValueError("indices must be non-empty")

        sample_indices = _normalize_indices(indices, self._length)
        batch = cast("TensorDict", self._dataset_td[sample_indices])
        _inject_lazy_readers(
            batch, idx=sample_indices, readers=self._feature_lazy_readers, namespace="features"
        )
        _inject_lazy_readers(
            batch, idx=sample_indices, readers=self._target_lazy_readers, namespace="targets"
        )
        return batch


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
