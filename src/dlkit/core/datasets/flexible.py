"""Flexible dataset implementation for arbitrary feature/target configurations.

This module provides FlexibleDataset that loads an arbitrary set of feature
and target files based on data entry configurations.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from dlkit.tools.config.data_entries import (
    DataEntry,
    FeatureType,
    PathBasedEntry,
    SparseFeature,
    TargetType,
    TensorDataEntry,
    ValueBasedEntry,
    to_tensor_entry,
)
from dlkit.tools.io import load_array
from dlkit.tools.io.sparse import (
    PackFiles,
    SparsePackReader,
    is_sparse_pack_dir,
    open_sparse_pack,
)

from .base import BaseDataset, register_dataset

if TYPE_CHECKING:
    from tensordict import TensorDict
    from tensordict.base import TensorDictBase


class BatchComplianceError(ValueError):
    """Raised when dataset entries violate batch-shape invariants.

    Enforces the contract that every entry must carry an explicit sample
    dimension N as its first axis, and all entries must agree on N.
    """


class PlaceholderNotResolvedError(ValueError):
    """Raised when a placeholder entry is used without value injection."""

    def __init__(self, entry_name: str) -> None:
        """Initialize with entry name.

        Args:
            entry_name: Name of the unresolved placeholder entry
        """
        super().__init__(
            f"Entry '{entry_name}' is a placeholder without path or value. "
            f"Either specify 'path' in config or inject 'value' programmatically."
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SparseSourceBinding:
    """Sparse source binding with read-time denormalization behavior."""

    reader: SparsePackReader
    source_path: Path  # pack directory path — for cache fingerprinting
    denormalize: bool = False


def _normalize_entries(
    entries: Any,
) -> dict[str, tuple[Path | Tensor | np.ndarray | SparseSourceBinding, str | None]]:
    """Extract path or value from DataEntry objects or pre-resolved tensor entries.

    Expects DataEntry objects (PathFeature, ValueFeature, PathTarget, ValueTarget)
    created by Feature() or Target() factories. These factories handle validation.

    Single Responsibility: Extract data sources from validated entries.
    No validation - trust that factories already validated.

    Args:
        entries: Collection of DataEntry objects

    Returns:
        Dictionary mapping entry name to tuple of (data source, entry name).
        The entry name is used as array_key when loading .npz files.

    Raises:
        TypeError: If receives dict (should use Feature()/Target() instead)
        PlaceholderNotResolvedError: If entry is placeholder without data
    """
    result: dict[str, tuple[Path | Tensor | np.ndarray | SparseSourceBinding, str | None]] = {}
    if entries is None:
        return result

    # Reject dicts - force users to use factories
    if isinstance(entries, dict):
        raise TypeError(
            "FlexibleDataset no longer accepts raw dicts. "
            "Use Feature() or Target() factories instead:\n"
            "  from dlkit.tools.config.data_entries import Feature, Target\n"
            "  features = [Feature(name='x', path='data.npy')]"
        )

    # list[DataEntry] entries
    for item in entries:
        # Already-resolved tensor entries
        if isinstance(item, TensorDataEntry):
            result[item.name] = (item.tensor, item.name)
            continue

        # Reject dicts in list
        if isinstance(item, dict):
            raise TypeError(
                "FlexibleDataset no longer accepts raw dicts. "
                "Use Feature(**dict) or Target(**dict) factories instead."
            )

        # ValueBasedEntry: extract in-memory value
        if isinstance(item, ValueBasedEntry):
            if item.is_placeholder():
                raise PlaceholderNotResolvedError(str(item.name or "unknown"))
            assert item.name is not None, "Non-placeholder entry must have name"
            result[item.name] = (item.value, item.name)  # type: ignore[assignment]

        # SparseFeature: open sparse pack reader
        elif isinstance(item, SparseFeature):
            if item.is_placeholder():
                raise PlaceholderNotResolvedError(str(item.name or "unknown"))
            assert item.name is not None, "Non-placeholder entry must have name"
            assert item.path is not None, "SparseFeature path must be set for non-placeholder entry"
            files = PackFiles(
                indices=item.files.indices,
                values=item.files.values,
                nnz_ptr=item.files.nnz_ptr,
                values_scale=item.files.values_scale,
            )
            reader = open_sparse_pack(Path(item.path), files=files)
            result[item.name] = (
                SparseSourceBinding(
                    reader=reader,
                    source_path=Path(item.path),
                    denormalize=item.denormalize,
                ),
                None,
            )

        # PathBasedEntry: extract file path
        elif isinstance(item, PathBasedEntry):
            if item.is_placeholder():
                raise PlaceholderNotResolvedError(str(item.name or "unknown"))
            assert item.name is not None, "Non-placeholder entry must have name"
            resolved_path = Path(item.path)  # type: ignore[arg-type]
            if is_sparse_pack_dir(resolved_path):
                result[item.name] = (
                    SparseSourceBinding(
                        reader=open_sparse_pack(resolved_path),
                        source_path=resolved_path,
                        denormalize=False,
                    ),
                    None,
                )
            else:
                result[item.name] = (resolved_path, item.name)

        # Generic DataEntry: check capabilities
        elif isinstance(item, DataEntry):
            if item.is_placeholder():
                raise PlaceholderNotResolvedError(str(item.name or "unknown"))
            assert item.name is not None, "Non-placeholder entry must have name"
            tensor_entry = to_tensor_entry(item)
            result[item.name] = (tensor_entry.tensor, item.name)

        else:
            raise TypeError(
                f"Unsupported entry type: {type(item).__name__}. "
                f"Expected DataEntry objects from Feature()/Target() factories."
            )

    return result


def _load_or_convert_tensor(
    source: Path | Tensor | np.ndarray,
    dtype: torch.dtype | None = None,
    array_key: str | None = None,
) -> Tensor:
    """Pure function: convert source to torch.Tensor with dtype handling.

    Handles both file paths (production) and in-memory data (testing).
    Respects precision context via PrecisionService for dtype resolution.

    Args:
        source: File path OR in-memory tensor/array
        dtype: Target dtype (uses PrecisionService if None)
        array_key: For .npz files, the array name to extract

    Returns:
        torch.Tensor with appropriate dtype
    """
    # Case 1: Already a torch.Tensor or numpy array (in-memory data)
    if isinstance(source, (torch.Tensor, np.ndarray)):
        tensor = torch.as_tensor(source)  # Zero-copy for numpy arrays

        # Apply dtype conversion if specified
        if dtype is not None:
            return tensor.to(dtype=dtype)

        # Use PrecisionService for dtype if not specified
        # This respects the global precision context set by precision_override()
        from dlkit.interfaces.api.services.precision_service import get_precision_service

        precision_service = get_precision_service()
        resolved_dtype = precision_service.get_torch_dtype()
        return tensor.to(dtype=resolved_dtype)

    # Case 2: File path - delegate to existing load_array()
    # load_array() already handles PrecisionService integration
    # For .npz files, pass array_key to select specific array
    if isinstance(source, Path) and source.suffix.lower() == ".npz":
        return load_array(source, dtype=dtype, array_key=array_key)
    return load_array(source, dtype=dtype)


def _load_sparse_tensor(
    reader: SparsePackReader,
    sample_index: int,
    *,
    denormalize: bool = False,
) -> Tensor:
    """Load one sparse tensor from a sparse pack reader."""
    return reader.build_torch_sparse(
        sample_index=sample_index,
        denormalize=denormalize,
        coalesce=False,
    )


def _get_source_dtype() -> torch.dtype:
    """Get the active precision dtype from PrecisionService.

    Returns:
        torch.dtype: Dtype from the active precision context.
    """
    from dlkit.interfaces.api.services.precision_service import get_precision_service

    return get_precision_service().get_torch_dtype()


def _source_to_fingerprint_path(name: str, source: Path | SparseSourceBinding) -> Path:
    """Pure function: extract the path used for mtime fingerprinting from a source.

    Args:
        name: Entry name (used in error message only).
        source: File path or sparse source binding.

    Returns:
        Path used for mtime-based cache invalidation.

    Raises:
        ValueError: If source is not a file-backed type.
    """
    match source:
        case Path():
            return source
        case SparseSourceBinding():
            return source.source_path
        case _:
            raise ValueError(
                f"Entry '{name}' is not file-backed. "
                "memmap_cache_dir requires PathBasedEntry or sparse pack entries."
            )


def _compute_source_fingerprint(
    all_maps: dict[str, tuple[Path | SparseSourceBinding, str | None]],
    dtype: torch.dtype,
) -> str:
    """Compute SHA-256 fingerprint over source files and dtype.

    Args:
        all_maps: Mapping from entry name to (source, array_key).
        dtype: Target tensor dtype included in the fingerprint.

    Returns:
        str: Full hex-digest fingerprint string.

    Raises:
        ValueError: If any source is not file-backed.
    """
    h = hashlib.sha256()
    for name in sorted(all_maps):
        source, _ = all_maps[name]
        fp_path = _source_to_fingerprint_path(name, source)
        h.update(name.encode())
        h.update(fp_path.as_posix().encode())
        h.update(str(fp_path.stat().st_mtime_ns).encode())
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


def _write_sparse_entry_to_memmap(
    name: str,
    binding: SparseSourceBinding,
    n_total: int,
    group_dir: Path,
    dtype: torch.dtype,
    chunk_size: int = 1_000,
) -> Any:
    """Densify a sparse pack into an (n_total, D, D) MemoryMappedTensor, chunk by chunk.

    Peak RAM is bounded to ``chunk_size × D × D × sizeof(dtype)`` regardless of
    ``n_total``.  The ``i % reader.n_samples`` indexing unifies broadcast packs
    (n_samples == 1) and normal packs without a branch.

    Args:
        name: Entry name (used for output filename).
        binding: Sparse source binding with reader and denormalization settings.
        n_total: Total rows to write.  May differ from ``reader.n_samples`` when
            the pack is a broadcast (n_samples == 1) expanded to a larger dataset.
        group_dir: Directory to write the ``.memmap`` file into.
        dtype: Target tensor dtype.
        chunk_size: Rows written per loop iteration.

    Returns:
        MemoryMappedTensor: Disk-backed tensor of shape ``(n_total, D, D)``.
    """
    from tensordict.memmap import MemoryMappedTensor

    reader = binding.reader
    d_rows, d_cols = reader.matrix_size
    filename = str(group_dir / f"{name}.memmap")
    mmt = MemoryMappedTensor.empty([n_total, d_rows, d_cols], dtype=dtype, filename=filename)

    total_start = perf_counter()
    if reader.n_samples == 1:
        build_dense_start = perf_counter()
        dense_matrix = reader.build_torch_sparse(
            sample_index=0,
            dtype=dtype,
            denormalize=binding.denormalize,
            coalesce=False,
        ).to_dense()
        build_dense_elapsed = perf_counter() - build_dense_start

        write_elapsed = 0.0
        dense_template = dense_matrix.unsqueeze(0)
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            write_start = perf_counter()
            mmt[start:end] = dense_template.expand(end - start, -1, -1)
            write_elapsed += perf_counter() - write_start

        logger.debug(
            "Sparse memmap write feature='{}' mode=broadcast n_total={} chunk_size={} "
            "dense_build_s={:.4f} write_s={:.4f} total_s={:.4f}",
            name,
            n_total,
            chunk_size,
            build_dense_elapsed,
            write_elapsed,
            perf_counter() - total_start,
        )
        return mmt

    sparse_build_elapsed = 0.0
    dense_build_elapsed = 0.0
    write_elapsed = 0.0
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        pack_indices = [i % reader.n_samples for i in range(start, end)]
        sparse_start = perf_counter()
        sparse_batch = reader.build_torch_sparse_batch(
            pack_indices,
            dtype=dtype,
            denormalize=binding.denormalize,
            coalesce=False,
        )
        sparse_build_elapsed += perf_counter() - sparse_start

        dense_start = perf_counter()
        dense_batch = torch.stack([s.to_dense() for s in sparse_batch])
        dense_build_elapsed += perf_counter() - dense_start

        write_start = perf_counter()
        mmt[start:end] = dense_batch
        write_elapsed += perf_counter() - write_start

    logger.debug(
        "Sparse memmap write feature='{}' mode=per_sample n_total={} chunk_size={} "
        "sparse_build_s={:.4f} dense_build_s={:.4f} write_s={:.4f} total_s={:.4f}",
        name,
        n_total,
        chunk_size,
        sparse_build_elapsed,
        dense_build_elapsed,
        write_elapsed,
        perf_counter() - total_start,
    )

    return mmt


def _determine_n_total(
    dense_tensors: dict[str, Any],
    sparse_bindings: dict[str, SparseSourceBinding],
) -> int:
    """Pure function: resolve the canonical sample count N from available tensors.

    Precedence: dense tensors (features + targets) → sparse reader n_samples > 1.

    Args:
        dense_tensors: Already-written tensors keyed by entry name.
        sparse_bindings: Sparse feature bindings not yet materialised.

    Returns:
        int: Canonical dataset size N.

    Raises:
        BatchComplianceError: If dense sizes disagree, or sparse n_samples disagree.
        ValueError: If N cannot be inferred (no dense, all-broadcast sparse).
    """
    dense_n_set = {int(t.shape[0]) for t in dense_tensors.values()}
    if dense_n_set:
        if len(dense_n_set) > 1:
            raise BatchComplianceError(
                f"All entries must share the same first dimension N, "
                f"but found differing sizes: {sorted(dense_n_set)}."
            )
        return next(iter(dense_n_set))

    if sparse_bindings:
        sparse_ns = {b.reader.n_samples for b in sparse_bindings.values() if b.reader.n_samples > 1}
        if not sparse_ns:
            raise ValueError(
                "Cannot infer dataset size: all sparse entries are broadcast packs "
                "(n_samples=1). Provide at least one non-broadcast entry or a dense target."
            )
        if len(sparse_ns) > 1:
            raise BatchComplianceError(
                f"Sparse entries have differing sample counts: {sorted(sparse_ns)}."
            )
        return next(iter(sparse_ns))

    raise ValueError("No entries provided to determine dataset size.")


def _validate_memmap_entries(
    all_maps: dict[str, tuple[Path | Tensor | np.ndarray | SparseSourceBinding, str | None]],
) -> None:
    """Raise ValueError if any entry is not file-backed.

    Args:
        all_maps: Mapping from entry name to (source, array_key).

    Raises:
        ValueError: If any source is not a Path or SparseSourceBinding.
    """
    for name, (source, _) in all_maps.items():
        if not isinstance(source, (Path, SparseSourceBinding)):
            raise ValueError(
                f"Entry '{name}' is not file-backed. "
                "memmap_cache_dir requires PathBasedEntry or sparse pack entries."
            )


def _load_dense_entries(
    feat_map: dict[str, tuple[Path | Tensor | np.ndarray | SparseSourceBinding, str | None]],
    targ_map: dict[str, tuple[Path | Tensor | np.ndarray | SparseSourceBinding, str | None]],
) -> tuple[dict[str, Tensor], dict[str, SparseSourceBinding], dict[str, Tensor]]:
    """Load dense tensors from entry maps; collect sparse feature bindings.

    Splits feat_map into dense tensors and deferred sparse bindings.
    Loads all target tensors, rejecting sparse targets.

    Args:
        feat_map: Feature entry map: name → (source, array_key).
        targ_map: Target entry map: name → (source, array_key).

    Returns:
        Tuple of (dense feature tensors, sparse bindings, target tensors).

    Raises:
        ValueError: If a target entry uses a sparse pack.
    """
    dense_feat_tensors: dict[str, Tensor] = {}
    sparse_bindings: dict[str, SparseSourceBinding] = {}
    for name, (source, array_key) in feat_map.items():
        if isinstance(source, SparseSourceBinding):
            sparse_bindings[name] = source
        else:
            dense_feat_tensors[name] = _load_or_convert_tensor(source, array_key=array_key)

    targ_tensors: dict[str, Tensor] = {}
    for name, (source, array_key) in targ_map.items():
        if isinstance(source, SparseSourceBinding):
            raise ValueError(f"Target entry '{name}' cannot use sparse packs.")
        targ_tensors[name] = _load_or_convert_tensor(source, array_key=array_key)

    return dense_feat_tensors, sparse_bindings, targ_tensors


def _resolve_n_and_materialize_sparse(
    dense_feat_tensors: dict[str, Tensor],
    targ_tensors: dict[str, Tensor],
    sparse_bindings: dict[str, SparseSourceBinding],
) -> tuple[dict[str, Tensor], int]:
    """Infer canonical N and materialize sparse features to dense tensors.

    Uses _determine_n_total for N resolution, then stacks each sparse pack
    into a (N, D, D) dense tensor.

    Args:
        dense_feat_tensors: Already-loaded dense feature tensors.
        targ_tensors: Target tensors.
        sparse_bindings: Sparse feature bindings not yet materialized.

    Returns:
        Tuple of (feature tensor map with sparse entries materialized, canonical N).

    Raises:
        BatchComplianceError: If sparse sample counts disagree with N, or stacking fails.
        ValueError: If N cannot be inferred from available entries.
    """
    if not sparse_bindings:
        return dict(dense_feat_tensors), 0

    # Filter out scalar tensors before N inference — scalar compliance is checked
    # by _assemble_dataset_tensordict; 0-D tensors have no shape[0].
    non_scalar_dense = {
        k: v for k, v in {**dense_feat_tensors, **targ_tensors}.items() if v.dim() > 0
    }
    n = _determine_n_total(non_scalar_dense, sparse_bindings)
    feat_tensors = dict(dense_feat_tensors)
    for name, binding in sparse_bindings.items():
        materialize_start = perf_counter()
        reader = binding.reader
        if reader.n_samples not in (1, n):
            raise BatchComplianceError(
                f"Sparse feature '{name}' has {reader.n_samples} samples but expected 1 or {n}."
            )
        try:
            if reader.n_samples == 1:
                shared_sparse = _load_sparse_tensor(reader, 0, denormalize=binding.denormalize)
                feat_tensors[name] = torch.stack([shared_sparse] * n)
                logger.debug(
                    "Sparse in-memory materialization feature='{}' mode=broadcast n_total={} "
                    "elapsed_s={:.4f}",
                    name,
                    n,
                    perf_counter() - materialize_start,
                )
            else:
                feat_tensors[name] = torch.stack(
                    [_load_sparse_tensor(reader, i, denormalize=binding.denormalize) for i in range(n)]
                )
                logger.debug(
                    "Sparse in-memory materialization feature='{}' mode=per_sample n_total={} "
                    "elapsed_s={:.4f}",
                    name,
                    n,
                    perf_counter() - materialize_start,
                )
        except RuntimeError as exc:
            raise BatchComplianceError(
                f"Failed to stack sparse feature '{name}'. Ensure sparse tensor collation support "
                "is available in current torch/tensordict versions."
            ) from exc
    return feat_tensors, n


def _validate_sparse_bindings_for_n(
    sparse_bindings: dict[str, SparseSourceBinding],
    n: int,
) -> None:
    """Validate that each sparse binding can align with resolved dataset size N."""
    for name, binding in sparse_bindings.items():
        n_sparse = binding.reader.n_samples
        if n_sparse not in (1, n):
            raise BatchComplianceError(
                f"Sparse feature '{name}' has {n_sparse} samples but expected 1 or {n}."
            )


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


def _inject_sparse_features(
    sample: TensorDict,
    *,
    sample_index: int,
    sparse_bindings: dict[str, SparseSourceBinding],
) -> TensorDict:
    """Return a sample TensorDict with sparse features injected on demand."""
    if not sparse_bindings:
        return sample

    for name, binding in sparse_bindings.items():
        reader = binding.reader
        sparse_tensor = reader.build_torch_sparse(
            sample_index=sample_index,
            denormalize=binding.denormalize,
            coalesce=False,
        )
        sample["features"][name] = sparse_tensor
    return sample


def _build_sparse_feature_batch(
    *,
    sample_indices: list[int],
    binding: SparseSourceBinding,
) -> Tensor:
    """Build a sparse feature batch for requested sample indices."""
    reader = binding.reader
    return reader.build_torch_sparse_stacked(
        sample_indices=sample_indices,
        denormalize=binding.denormalize,
        coalesce=False,
    )


def _inject_sparse_features_batch(
    batch: TensorDict,
    *,
    sample_indices: list[int],
    sparse_bindings: dict[str, SparseSourceBinding],
) -> TensorDict:
    """Inject sparse features into a batched TensorDict."""
    if not sparse_bindings:
        return batch

    for name, binding in sparse_bindings.items():
        batch["features"][name] = _build_sparse_feature_batch(
            sample_indices=sample_indices,
            binding=binding,
        )
    return batch


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
    from tensordict import TensorDict

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

    feature_batch = {name: feat_tensors[name] for name in feat_names}
    target_batch = {name: targ_tensors[name] for name in targ_names}
    td = TensorDict(
        {
            "features": TensorDict(feature_batch, batch_size=[n]),
            "targets": TensorDict(target_batch, batch_size=[n]),
        },
        batch_size=[n],
    )
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
    feat_map: dict[str, tuple[Path | SparseSourceBinding, str | None]],
    targ_map: dict[str, tuple[Path, str | None]],
    cache_dir: Path,
    dtype: torch.dtype,
    chunk_size: int = 5_000,
) -> TensorDict:
    """Build an OOM-safe memmap cache for a dataset.

    Dense and target entries are written first so N can be resolved before
    sparse entries are densified.  Sparse entries are written in a second pass
    with ``n_total=N``, bounding peak RAM to ``chunk_size × D × D`` rows.

    Args:
        feat_map: Feature entry map: name → (source, array_key).  Source may be
            a ``Path`` (dense) or a ``SparseSourceBinding`` (sparse-to-dense).
        targ_map: Target entry map: name → (source_path, array_key).
        cache_dir: Root directory for the cache.
        dtype: Target tensor dtype for all entries.
        chunk_size: Rows per write iteration.

    Returns:
        TensorDict: Dataset TensorDict backed by the new memmap files.

    Raises:
        BatchComplianceError: If feature/target sample counts disagree.
        ValueError: If N cannot be inferred from available entries.
    """
    from tensordict import TensorDict

    features_dir = cache_dir / "features"
    targets_dir = cache_dir / "targets"
    for d in (cache_dir, features_dir, targets_dir):
        d.mkdir(parents=True, exist_ok=True)

    meta: dict[str, dict[str, Any]] = {"features": {}, "targets": {}}
    feat_tensors: dict[str, Any] = {}
    targ_tensors: dict[str, Any] = {}
    sparse_feat_bindings: dict[str, SparseSourceBinding] = {}

    # First pass: write dense feature entries; collect sparse bindings for later.
    for name, (source, array_key) in feat_map.items():
        match source:
            case Path():
                mmt = _write_entry_to_memmap(name, source, array_key, features_dir, dtype, chunk_size)
                feat_tensors[name] = mmt
                meta["features"][name] = {"shape": list(mmt.shape), "dtype": str(dtype)}
            case SparseSourceBinding():
                sparse_feat_bindings[name] = source

    # Targets are always dense.
    for name, (source, array_key) in targ_map.items():
        mmt = _write_entry_to_memmap(name, source, array_key, targets_dir, dtype, chunk_size)
        targ_tensors[name] = mmt
        meta["targets"][name] = {"shape": list(mmt.shape), "dtype": str(dtype)}

    # Resolve N (pure): dense entries take precedence; fall back to sparse readers.
    n = _determine_n_total({**feat_tensors, **targ_tensors}, sparse_feat_bindings)

    # Second pass: densify sparse features now that N is known.
    for name, binding in sparse_feat_bindings.items():
        mmt = _write_sparse_entry_to_memmap(name, binding, n, features_dir, dtype, chunk_size)
        feat_tensors[name] = mmt
        meta["features"][name] = {"shape": list(mmt.shape), "dtype": str(dtype)}

    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    return TensorDict(
        {
            "features": TensorDict(feat_tensors, batch_size=[n]),
            "targets": TensorDict(targ_tensors, batch_size=[n]),
        },
        batch_size=[n],
    )


def _load_memmap_from_cache(cache_dir: Path) -> TensorDict:
    """Reconstruct a TensorDict from an existing memmap cache directory.

    Args:
        cache_dir: Cache directory previously built by ``_build_memmap_cache``.

    Returns:
        TensorDict: Dataset TensorDict backed by memory-mapped files.
    """
    from tensordict import TensorDict

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
    return TensorDict(
        {
            "features": TensorDict(feat_tensors, batch_size=[n]),
            "targets": TensorDict(targ_tensors, batch_size=[n]),
        },
        batch_size=[n],
    )


def _load_or_build_memmap(
    feat_map: dict[str, tuple[Path | SparseSourceBinding, str | None]],
    targ_map: dict[str, tuple[Path, str | None]],
    cache_dir: Path,
    dtype: torch.dtype,
    chunk_size: int = 5_000,
) -> TensorDict:
    """Load a memmap dataset from cache, rebuilding if stale or absent.

    Cache invalidation is fingerprint-based: any change to source file mtimes
    or dtype triggers a full wipe-and-rebuild (no stale sub-directories).

    Args:
        feat_map: Feature entry map: name → (source, array_key).
        targ_map: Target entry map: name → (source_path, array_key).
        cache_dir: Directory for memmap cache files.
        dtype: Target tensor dtype (included in fingerprint).
        chunk_size: Rows written per iteration when building the cache.

    Returns:
        TensorDict: Dataset TensorDict backed by memory-mapped files.
    """
    all_maps: dict[str, tuple[Path | SparseSourceBinding, str | None]] = {
        **feat_map,  # type: ignore[arg-type]
        **targ_map,
    }
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
class FlexibleDataset(BaseDataset):
    """Dataset that loads an arbitrary set of feature and target files.

    Entries are provided as DataEntry objects created via Feature() or Target()
    factories. The key used in __getitem__ output is the entry name, and the
    value is the tensor slice at the requested index.

    Precision handling is automatic via the global precision service. Use
    precision_override() context to control the dtype of loaded tensors.

    Supported File Formats:
    - NumPy arrays: .npy (single array), .npz (multi-array)
    - PyTorch tensors: .pt, .pth
    - Text files: .txt, .csv

    For .npz files with multiple arrays, the entry name is used as the array key
    to select which array to load from the file.

    Supports:
    - Path-based entries: Data loaded from files (PathFeature, PathTarget)
    - Value-based entries: In-memory data (ValueFeature, ValueTarget)
    - Placeholder entries: Must be resolved before use (raises PlaceholderNotResolvedError)

    Single Responsibility: Load and manage dataset lifecycle (NO validation).
    Validation is handled by Feature()/Target() factories.

    Examples:
        Basic usage with .npy files:
            >>> from dlkit.tools.config.data_entries import Feature, Target
            >>> features = [Feature(name="x", path="data.npy")]
            >>> targets = [Target(name="y", path="labels.npy")]
            >>> dataset = FlexibleDataset(features=features, targets=targets)

        Using .npz files (entry name used as array key):
            >>> features = [Feature(name="features", path="data.npz")]
            >>> targets = [Target(name="targets", path="data.npz")]
            >>> dataset = FlexibleDataset(features=features, targets=targets)
            # Loads array "features" and "targets" from data.npz

        Multiple features from same .npz file:
            >>> features = [
            ...     Feature(name="features", path="data.npz"),
            ...     Feature(name="latent", path="data.npz"),
            ... ]
            >>> dataset = FlexibleDataset(features=features)

        Mixed file formats:
            >>> features = [
            ...     Feature(name="x", path="features.npy"),
            ...     Feature(name="y", path="extra.npz"),  # Uses "y" as array key
            ... ]
            >>> dataset = FlexibleDataset(features=features)
    """

    def __init__(
        self,
        *,
        features: Iterable[FeatureType],
        targets: Iterable[TargetType] | None = None,
        memmap_cache_dir: Path | None = None,
        memmap_chunk_size: int = 5_000,
    ) -> None:
        """Initialize FlexibleDataset with feature and target entries.

        Args:
            features: Feature entries (PathFeature, ValueFeature, or SparseFeature).
            targets: Target entries (PathTarget or ValueTarget from Target() factory).
            memmap_cache_dir: If set, load dataset via OS memory-mapped files stored in
                this directory.  Entries must be file-backed (PathBasedEntry or sparse
                pack entries).  The cache is invalidated when source files or dtype change.
            memmap_chunk_size: Rows written per iteration when building the memmap cache.
                Bounds peak RAM to ``chunk_size × feature_width × sizeof(dtype)``.
                Ignored when ``memmap_cache_dir`` is ``None``.

        Raises:
            BatchComplianceError: If any entry is scalar (0-D) or N sizes do not agree.
            ValueError: If no features or targets are provided, or a non-file-backed
                entry is used with memmap_cache_dir.
            PlaceholderNotResolvedError: If placeholder entry without value.
            TypeError: If raw dicts are passed (use Feature()/Target() instead).
        """
        feat_map = _normalize_entries(features)
        targ_map = _normalize_entries(targets)

        if not feat_map and not targ_map:
            raise ValueError("At least one feature or target entry is required")

        self._feature_names: tuple[str, ...] = tuple(feat_map.keys())
        self._target_names: tuple[str, ...] = tuple(targ_map.keys())
        self._sparse_bindings: dict[str, SparseSourceBinding] = {}

        if memmap_cache_dir is not None:
            _validate_memmap_entries({**feat_map, **targ_map})
            dtype = _get_source_dtype()
            self._dataset_td = _load_or_build_memmap(
                feat_map,  # type: ignore[arg-type]
                targ_map,  # type: ignore[arg-type]
                Path(memmap_cache_dir),
                dtype,
                memmap_chunk_size,
            )
        else:
            dense_feats, sparse_bindings, targs = _load_dense_entries(feat_map, targ_map)
            self._sparse_bindings = sparse_bindings

            # Resolve canonical N using dense entries first, sparse bindings as fallback.
            non_scalar_dense = {k: v for k, v in {**dense_feats, **targs}.items() if v.dim() > 0}
            n = _determine_n_total(non_scalar_dense, sparse_bindings)
            _validate_sparse_bindings_for_n(sparse_bindings, n)

            if dense_feats or targs:
                self._dataset_td, _ = _assemble_dataset_tensordict(
                    dense_feats,
                    targs,
                    tuple(dense_feats.keys()),
                    self._target_names,
                )
            else:
                # Sparse-only dataset path: keep an empty dense base and inject sparse per sample.
                self._dataset_td = _make_empty_dataset_tensordict(n)

        self._length = int(self._dataset_td.batch_size[0])

    def __len__(self) -> int:
        """Return number of samples in dataset.

        Returns:
            Number of samples (first dimension of tensors)
        """
        return self._length

    def __getitem__(self, idx: int) -> TensorDict:
        """Get sample at index.

        Args:
            idx: Sample index

        Returns:
            TensorDict with feature and target nested TensorDicts (batch_size=[])
        """
        sample_index = _normalize_index(idx, self._length)
        sample = self._dataset_td[sample_index]
        return _inject_sparse_features(
            sample,
            sample_index=sample_index,
            sparse_bindings=self._sparse_bindings,
        )

    def __getitems__(self, indices: list[int]) -> TensorDict:
        """Get a batched TensorDict for a list of indices.

        This path is used by PyTorch DataLoader for map-style datasets when
        auto-collation is enabled, allowing sparse feature injection once per batch.
        """
        if not indices:
            raise ValueError("indices must be non-empty")

        sample_indices = _normalize_indices(indices, self._length)
        batch = self._dataset_td[sample_indices]
        return _inject_sparse_features_batch(
            batch,
            sample_indices=sample_indices,
            sparse_bindings=self._sparse_bindings,
        )


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
    from tensordict import TensorDictBase, stack as td_stack

    if isinstance(batch, TensorDictBase):
        return batch

    result = td_stack(batch, dim=0)
    if result.batch_size[0] != len(batch):
        raise BatchComplianceError(
            f"Collated batch size {result.batch_size[0]} does not match expected {len(batch)}."
        )
    return result
