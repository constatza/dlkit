"""Sparse pack validation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._manifest import PackFiles, PackManifest, _manifest_from_arrays
from ._protocols import SparseFormat
from ._registry import get_codec


def _validate_coo_pack(
    *,
    indices: np.ndarray,
    values: np.ndarray,
    nnz_ptr: np.ndarray,
    value_scale_from_disk: float,
    manifest: PackManifest,
) -> None:
    """Validate COO sparse pack payload against a manifest (pure, no I/O).

    Args:
        indices: Loaded COO indices array of shape (2, total_nnz).
        values: Loaded COO values array of shape (total_nnz,).
        nnz_ptr: Loaded row pointer array of shape (n_samples + 1,).
        value_scale_from_disk: Scale loaded from the pack's ``values_scale.npy``.
        manifest: Contract to validate the payload against.

    Raises:
        ValueError: On any shape mismatch, pointer inconsistency, dtype conflict,
            bounds violation, or value_scale mismatch.
    """
    if indices.ndim != 2 or indices.shape[0] != 2:
        raise ValueError(f"indices must have shape (2, total_nnz), got {indices.shape}")
    if values.ndim != 1:
        raise ValueError(f"values must be 1D, got {values.shape}")
    if nnz_ptr.ndim != 1:
        raise ValueError(f"nnz_ptr must be 1D, got {nnz_ptr.shape}")
    if nnz_ptr.size != manifest.n_samples + 1:
        raise ValueError(
            f"nnz_ptr size ({nnz_ptr.size}) must equal n_samples + 1 ({manifest.n_samples + 1})"
        )
    if nnz_ptr[0] != 0:
        raise ValueError("nnz_ptr must start at 0")
    if np.any(np.diff(nnz_ptr) < 0):
        raise ValueError("nnz_ptr must be non-decreasing")
    if int(nnz_ptr[-1]) != values.size:
        raise ValueError(
            f"nnz_ptr last value ({int(nnz_ptr[-1])}) must equal total nnz ({values.size})"
        )
    if indices.shape[1] != values.size:
        raise ValueError(
            f"indices nnz ({indices.shape[1]}) does not match values nnz ({values.size})"
        )
    if manifest.total_nnz != values.size:
        raise ValueError(
            f"manifest total_nnz ({manifest.total_nnz}) does not match values nnz ({values.size})"
        )
    if not np.isfinite(manifest.value_scale) or manifest.value_scale <= 0.0:
        raise ValueError(f"value_scale must be finite and > 0, got {manifest.value_scale}")
    if not np.isclose(value_scale_from_disk, manifest.value_scale):
        raise ValueError(
            f"value_scale mismatch: payload ({value_scale_from_disk}) != contract ({manifest.value_scale})"
        )

    manifest_dtype = np.dtype(manifest.dtype)
    if values.dtype != manifest_dtype:
        raise ValueError(
            f"manifest dtype ({manifest_dtype.name}) does not match values dtype ({values.dtype.name})"
        )

    rows, cols = manifest.matrix_size
    row_idx = indices[0]
    col_idx = indices[1]
    if np.any(row_idx < 0) or np.any(row_idx >= rows):
        raise ValueError(f"row indices must be in [0, {rows}), got out-of-bounds entries")
    if np.any(col_idx < 0) or np.any(col_idx >= cols):
        raise ValueError(f"column indices must be in [0, {cols}), got out-of-bounds entries")


def validate_sparse_pack(
    path: Path,
    *,
    format: SparseFormat = SparseFormat.COO,
    manifest: PackManifest | None = None,
    files: PackFiles | None = None,
    matrix_size: tuple[int, int] | None = None,
    dtype: np.dtype | str | None = None,
) -> None:
    """Validate sparse pack payload arrays directly from directory files.

    Each file is loaded exactly once.  If ``manifest`` is provided it is used as
    the authoritative contract; otherwise one is inferred from the payload.

    Args:
        path: Pack directory to validate.
        format: Storage format; drives codec selection via the registry.
        manifest: Authoritative contract; validated against payload when given.
        files: Custom payload filenames; ignored if ``manifest`` is provided.
        matrix_size: Explicit matrix dimensions for manifest inference.
        dtype: Explicit dtype for manifest inference.

    Raises:
        ValueError: On any payload inconsistency or manifest mismatch.
    """
    codec = get_codec(format)
    resolved_files = manifest.files if manifest is not None else (files or PackFiles())

    # Load once — no double-load
    indices, values, nnz_ptr = codec.load_arrays(path, resolved_files)
    payload_scale = codec.load_value_scale(path, resolved_files)

    # When a manifest is provided, its scale is the contract to validate against.
    # The disk scale (payload_scale) is always used for the integrity check.
    value_scale = float(manifest.value_scale) if manifest is not None else payload_scale
    resolved_matrix_size = manifest.matrix_size if manifest is not None else matrix_size
    resolved_dtype = manifest.dtype if manifest is not None else dtype

    inferred_manifest = _manifest_from_arrays(
        indices=indices,
        values=values,
        nnz_ptr=nnz_ptr,
        files=resolved_files,
        matrix_size=resolved_matrix_size,
        dtype=resolved_dtype,
        value_scale=value_scale,
    )

    if manifest is not None:
        if manifest.format != format:
            raise ValueError(
                f"validate_sparse_pack expected {format.value} manifest, "
                f"got '{manifest.format.value}'"
            )
        if manifest.n_samples != inferred_manifest.n_samples:
            raise ValueError(
                f"manifest n_samples ({manifest.n_samples}) does not match payload "
                f"({inferred_manifest.n_samples})"
            )
        if manifest.total_nnz != inferred_manifest.total_nnz:
            raise ValueError(
                f"manifest total_nnz ({manifest.total_nnz}) does not match payload "
                f"({inferred_manifest.total_nnz})"
            )

    _validate_coo_pack(
        indices=indices,
        values=values,
        nnz_ptr=nnz_ptr,
        value_scale_from_disk=payload_scale,
        manifest=inferred_manifest,
    )
