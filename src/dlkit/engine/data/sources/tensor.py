"""In-memory tensor source.

Wraps an already-resolved ``torch.Tensor`` as an ``ArraySource`` so that
programmatic or pre-loaded tensors integrate with the same data pipeline
as file-backed sources.
"""

from __future__ import annotations

import torch

from dlkit.infrastructure.precision.service import PrecisionService


class TensorSource:
    """Wraps an already-resolved in-memory ``Tensor`` as an ``ArraySource``.

    Use this when data has already been loaded or constructed in memory and
    you want to feed it through the same pipeline as file-backed sources.

    Args:
        data: A ``torch.Tensor`` whose leading dimension is the sample axis.

    Example:
        >>> import torch
        >>> src = TensorSource(torch.randn(50, 16))
        >>> src.n_samples
        50
        >>> src.get_item(3).shape
        torch.Size([16])
        >>> src.get_batch([0, 1, 2]).shape
        torch.Size([3, 16])
    """

    def __init__(self, data: torch.Tensor) -> None:
        """Wrap ``data`` as a per-sample source.

        Args:
            data: Tensor of shape ``(N, *sample_shape)`` where ``N`` is the
                  number of samples.
        """
        self._data = data

    @property
    def n_samples(self) -> int:
        """Number of samples along the leading axis.

        Returns:
            Integer length of dimension ``0`` of the wrapped tensor.
        """
        return int(self._data.shape[0])

    def get_item(self, idx: int) -> torch.Tensor:
        """Return a single sample tensor.

        Args:
            idx: Zero-based sample index.

        Returns:
            Tensor of shape ``(*sample_shape,)``.
        """
        return PrecisionService().cast_tensor(self._data[idx])

    def get_batch(self, indices: list[int]) -> torch.Tensor:
        """Return a batch tensor for the given indices.

        Args:
            indices: List of zero-based sample indices.

        Returns:
            Tensor of shape ``(B, *sample_shape)`` where ``B = len(indices)``.
        """
        return PrecisionService().cast_tensor(self._data[indices])


__all__ = ["TensorSource"]
