"""Eager file-based array source.

Loads the entire array from disk at construction time and serves slices
from an in-memory ``torch.Tensor``.  Suitable for datasets that fit in
RAM and benefit from zero per-item I/O overhead during training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dlkit.infrastructure.io import load_array


class EagerFileSource:
    """Loads an array from disk at ``__init__`` time; slices in-memory thereafter.

    The defining axis is *load strategy* (eager on construction), not file
    format.  Supports ``.npy``, ``.npz``, ``.csv``, ``.pt``, ``.pth`` via
    :func:`dlkit.infrastructure.io.load_array`.

    Args:
        path: Path to the data file.
        dtype: Optional ``torch.dtype`` override.  When ``None`` the global
               precision service resolves the dtype automatically.
        array_key: Key used to select an array from multi-array formats
                   (e.g. the named array inside a ``.npz`` archive).
        **load_kwargs: Additional keyword arguments forwarded to ``load_array``
                       (e.g. ``mmap_mode='r'`` for NumPy memory-mapped files).

    Example:
        >>> import numpy as np, torch
        >>> from pathlib import Path
        >>> # (tmp_path is provided by pytest)
        >>> p = tmp_path / "x.npy"
        >>> np.save(p, np.ones((100, 8), dtype="float32"))
        >>> src = EagerFileSource(p)
        >>> src.n_samples
        100
        >>> src.get_item(0).shape
        torch.Size([8])
    """

    def __init__(
        self,
        path: Path,
        dtype: torch.dtype | None = None,
        array_key: str | None = None,
        **load_kwargs: Any,
    ) -> None:
        """Load the array from ``path`` into memory.

        Args:
            path: Path to the data file.
            dtype: Optional dtype override; ``None`` defers to precision service.
            array_key: Array key for multi-array formats such as ``.npz``.
            **load_kwargs: Forwarded verbatim to ``load_array``.
        """
        kwargs: dict[str, Any] = {**load_kwargs}
        if array_key is not None:
            kwargs["array_key"] = array_key
        self._data: torch.Tensor = load_array(path, dtype=dtype, **kwargs)

    @property
    def n_samples(self) -> int:
        """Number of samples along the leading axis.

        Returns:
            Integer length of dimension ``0`` of the loaded tensor.
        """
        return int(self._data.shape[0])

    def get_item(self, idx: int) -> torch.Tensor:
        """Return a single sample tensor.

        Args:
            idx: Zero-based sample index.

        Returns:
            Tensor of shape ``(*sample_shape,)``.
        """
        return self._data[idx]

    def get_batch(self, indices: list[int]) -> torch.Tensor:
        """Return a batch tensor for the given indices.

        Args:
            indices: List of zero-based sample indices.

        Returns:
            Tensor of shape ``(B, *sample_shape)`` where ``B = len(indices)``.
        """
        return self._data[indices]


__all__ = ["EagerFileSource"]
