"""Manifest contracts, JSON I/O, and manifest-building helpers for sparse packs."""

from __future__ import annotations

import json
from dataclasses import field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from pydantic import ConfigDict, NonNegativeInt, PositiveInt, TypeAdapter
from pydantic.dataclasses import dataclass as pydantic_dataclass

from ._protocols import SparseFormat

_MANIFEST_FILENAME = "manifest.json"


def _normalize_value_scale(value_scale: float) -> float:
    """Validate and normalize value scale.

    Args:
        value_scale: The scale value to validate.

    Returns:
        The validated scale as a Python float.

    Raises:
        ValueError: If scale is not finite or not > 0.
    """
    scale = float(value_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"Invalid value_scale: {value_scale}. Expected finite value > 0."
        )
    return scale


def _infer_matrix_size(
    indices: np.ndarray,
    matrix_size: tuple[int, int] | None,
) -> tuple[int, int]:
    """Infer matrix size from COO indices when not explicitly provided.

    Args:
        indices: COO indices array of shape (2, nnz).
        matrix_size: Explicit size override; returned as-is if not None.

    Returns:
        Inferred or provided (rows, cols) matrix size.

    Raises:
        ValueError: If indices are empty and no explicit size is provided.
    """
    if matrix_size is not None:
        return matrix_size
    if indices.shape[1] == 0:
        raise ValueError(
            "Cannot infer matrix_size from empty indices. Provide matrix_size explicitly."
        )
    rows = int(indices[0].max()) + 1
    cols = int(indices[1].max()) + 1
    return rows, cols

# Schema identifiers are explicit and versioned. Add new schema dataclasses for
# new formats/versions and register them in _MANIFEST_ADAPTERS.
COO_PACK_SCHEMA = "dlkit.sparse-pack.coo.v1"


def _validate_payload_filename(name: str, field_name: str) -> None:
    """Validate one manifest-declared payload filename."""
    if not name:
        raise ValueError(f"{field_name} filename must be non-empty")
    if "/" in name or "\\" in name:
        raise ValueError(f"{field_name} filename must be a local basename, got '{name}'")
    if not name.endswith(".npy"):
        raise ValueError(f"{field_name} filename must end with '.npy', got '{name}'")


@pydantic_dataclass(config=ConfigDict(frozen=True))
class CooPackFiles:
    """COO payload naming contract."""

    indices: str = "indices.npy"
    values: str = "values.npy"
    nnz_ptr: str = "nnz_ptr.npy"
    values_scale: str = "values_scale.npy"

    def __post_init__(self) -> None:
        _validate_payload_filename(self.indices, "indices")
        _validate_payload_filename(self.values, "values")
        _validate_payload_filename(self.nnz_ptr, "nnz_ptr")
        _validate_payload_filename(self.values_scale, "values_scale")


@pydantic_dataclass(config=ConfigDict(frozen=True))
class CooPackManifest:
    """Schema contract for COO sparse packs."""

    schema: Literal["dlkit.sparse-pack.coo.v1"] = COO_PACK_SCHEMA
    format: Literal[SparseFormat.COO] = SparseFormat.COO
    n_samples: PositiveInt = 1
    matrix_size: tuple[PositiveInt, PositiveInt] = (1, 1)
    dtype: str = "float32"
    total_nnz: NonNegativeInt = 0
    value_scale: float = 1.0
    files: CooPackFiles = field(default_factory=CooPackFiles)

    def __post_init__(self) -> None:
        _ = self.files
        try:
            np.dtype(self.dtype)
        except Exception as exc:
            raise ValueError(f"Invalid dtype in sparse pack manifest: {self.dtype}") from exc
        if not np.isfinite(self.value_scale) or self.value_scale <= 0.0:
            raise ValueError(
                f"Invalid value_scale in sparse pack manifest: {self.value_scale}. "
                "Expected finite value > 0."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to JSON-compatible dictionary."""
        return _COO_MANIFEST_ADAPTER.dump_python(self, mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CooPackManifest:
        """Deserialize dictionary to validated COO manifest."""
        try:
            return _COO_MANIFEST_ADAPTER.validate_python(data)
        except Exception as exc:
            raise ValueError(f"Invalid sparse pack manifest: {data}") from exc


_COO_MANIFEST_ADAPTER = TypeAdapter(CooPackManifest)

# Backward-compatible names used by existing callers.
PackFiles: TypeAlias = CooPackFiles
PackManifest: TypeAlias = CooPackManifest
SparsePackManifest: TypeAlias = CooPackManifest


def _manifest_from_arrays(
    *,
    indices: np.ndarray,
    values: np.ndarray,
    nnz_ptr: np.ndarray,
    files: CooPackFiles,
    matrix_size: tuple[int, int] | None = None,
    dtype: np.dtype | str | None = None,
    value_scale: float = 1.0,
) -> CooPackManifest:
    """Build a validated COO manifest from loaded arrays and optional overrides.

    Args:
        indices: COO indices array of shape (2, total_nnz).
        values: COO values array of shape (total_nnz,).
        nnz_ptr: Row pointer array of shape (n_samples + 1,).
        files: Payload filename contract.
        matrix_size: Explicit matrix dimensions; inferred from indices if None.
        dtype: Explicit dtype override; inferred from values if None.
        value_scale: Normalization scale factor; must be finite and > 0.

    Returns:
        Validated ``CooPackManifest`` reflecting the actual array contents.

    Raises:
        ValueError: If nnz_ptr is malformed or value_scale is invalid.
    """
    if nnz_ptr.ndim != 1 or nnz_ptr.size < 2:
        raise ValueError(f"nnz_ptr must be 1D with at least 2 entries, got {nnz_ptr.shape}")
    inferred_n_samples = int(nnz_ptr.size - 1)
    resolved_dtype = np.dtype(dtype or values.dtype).name
    resolved_size = _infer_matrix_size(indices, matrix_size)
    resolved_scale = _normalize_value_scale(value_scale)
    return CooPackManifest(
        format=SparseFormat.COO,
        n_samples=inferred_n_samples,
        matrix_size=(int(resolved_size[0]), int(resolved_size[1])),
        dtype=resolved_dtype,
        total_nnz=int(values.size),
        value_scale=resolved_scale,
        files=files,
    )


_MANIFEST_ADAPTERS: dict[str, TypeAdapter[Any]] = {
    COO_PACK_SCHEMA: _COO_MANIFEST_ADAPTER,
}


def register_manifest_schema(schema: str, adapter: TypeAdapter[Any]) -> None:
    """Register a new sparse pack manifest schema adapter."""
    if not schema:
        raise ValueError("schema must be non-empty")
    _MANIFEST_ADAPTERS[schema] = adapter


def _dump_manifest(manifest: SparsePackManifest) -> dict[str, Any]:
    """Dump manifest instance using its registered schema adapter."""
    adapter = _MANIFEST_ADAPTERS.get(manifest.schema)
    if adapter is None:
        raise ValueError(f"Unregistered sparse pack schema: {manifest.schema}")
    return adapter.dump_python(manifest, mode="json")


def _load_manifest(data: dict[str, Any]) -> SparsePackManifest:
    """Load manifest dictionary via schema-dispatch."""
    schema = data.get("schema")
    if not isinstance(schema, str):
        raise ValueError("Sparse pack manifest must include string field 'schema'")
    adapter = _MANIFEST_ADAPTERS.get(schema)
    if adapter is None:
        known = ", ".join(sorted(_MANIFEST_ADAPTERS))
        raise ValueError(f"Unsupported sparse pack schema '{schema}'. Known: {known}")
    try:
        return adapter.validate_python(data)
    except Exception as exc:
        raise ValueError(f"Invalid sparse pack manifest: {data}") from exc


def get_manifest_path(path: Path) -> Path:
    """Get manifest path for a sparse pack directory."""
    return Path(path) / _MANIFEST_FILENAME


def has_manifest(path: Path) -> bool:
    """Check whether a sparse pack manifest exists."""
    return get_manifest_path(path).exists()


def read_manifest(path: Path) -> SparsePackManifest:
    """Read and validate sparse pack manifest from a directory."""
    manifest_path = get_manifest_path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Sparse pack manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in sparse pack manifest: {manifest_path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Sparse pack manifest must be an object: {manifest_path}")
    return _load_manifest(data)


def write_manifest(path: Path, manifest: SparsePackManifest) -> None:
    """Write sparse pack manifest to a directory."""
    pack_dir = Path(path)
    pack_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pack_dir / _MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(_dump_manifest(manifest), handle, indent=2, sort_keys=True)
