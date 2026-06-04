"""Manifest contract for dense array packs."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, PositiveInt

ARRAY_PACK_SCHEMA = "dlkit.array-pack.zarr-dense.v1"
"""Versioned schema identifier for zarr-dense array packs."""


class ArrayPackManifest(BaseModel, frozen=True):
    """Human-readable schema contract stored inside a zarr-dense array pack.

    Written as ``group.attrs["dlkit_manifest"]`` at pack close-time and read
    back by ``ZarrDensePackReader`` to recover shape and dtype metadata without
    inspecting individual array headers.

    Attributes:
        schema_: Versioned schema identifier; always
            ``"dlkit.array-pack.zarr-dense.v1"``.
        n_samples: Number of matrices stored in the pack.
        matrix_size: Shared ``(rows, cols)`` dimensions for every sample.
        dtype: NumPy dtype name for the data array (e.g. ``"float32"``).
        chunk_size: Number of samples per zarr chunk along the sample axis.

    Raises:
        ValueError: On construction if ``dtype`` is not a valid numpy dtype
            name or if ``matrix_size`` or ``n_samples`` are non-positive.

    Example::

        manifest = ArrayPackManifest(
            n_samples=100,
            matrix_size=(64, 64),
            dtype="float32",
            chunk_size=64,
        )
        group.attrs["dlkit_manifest"] = manifest.model_dump(by_alias=True)
    """

    schema_: Literal["dlkit.array-pack.zarr-dense.v1"] = Field(ARRAY_PACK_SCHEMA, alias="schema")
    n_samples: PositiveInt
    matrix_size: tuple[PositiveInt, PositiveInt]
    dtype: str
    chunk_size: PositiveInt

    def model_post_init(self, __context: Any) -> None:
        """Validate dtype after construction.

        Args:
            __context: Pydantic internal context (unused).

        Raises:
            ValueError: If ``dtype`` is not a valid numpy dtype name.
        """
        try:
            np.dtype(self.dtype)
        except Exception as exc:
            raise ValueError(f"Invalid dtype in array pack manifest: {self.dtype}") from exc

    @classmethod
    def from_attrs(cls, attrs: dict[str, Any]) -> ArrayPackManifest:
        """Deserialize a manifest from zarr group attributes.

        Args:
            attrs: Raw attribute dictionary read from ``group.attrs``.

        Returns:
            Validated ``ArrayPackManifest``.

        Raises:
            ValueError: If the attributes do not conform to the manifest schema.
        """
        try:
            return cls.model_validate(attrs, from_attributes=False)
        except Exception as exc:
            raise ValueError(f"Invalid array pack manifest attrs: {attrs}") from exc
