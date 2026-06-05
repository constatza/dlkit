"""Resolved data source produced by DataEntry.normalize()."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from torch import Tensor

if TYPE_CHECKING:
    from dlkit.infrastructure.zarr import ILazyReader


@dataclass(frozen=True)
class NormalizedEntry:
    """Normalized entry after source extraction.

    Attributes:
        source: Data source — ILazyReader for lazy arrays, Path for
            file-backed entries, or Tensor/ndarray for in-memory data.
        array_key: Array key for .npz files; equals entry name by convention.
            None for non-npz sources.
        load_kwargs: Extra keyword arguments forwarded to load_array().
    """

    source: ILazyReader | Path | Tensor | np.ndarray
    array_key: str | None
    load_kwargs: dict[str, Any] = dc_field(default_factory=dict)
