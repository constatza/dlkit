"""Base source types and type aliases for the data source layer.

Defines ``NamedSources``, an ordered immutable association list, and
``BroadcastSource``, which replicates a single-sample tensor across any
requested batch size.
"""

from __future__ import annotations

import torch

from dlkit.common.sources import ArraySource

# Ordered association list — preserves insertion order, fully immutable.
type NamedSources = tuple[tuple[str, ArraySource], ...]
"""Immutable ordered sequence of ``(name, ArraySource)`` pairs."""


class BroadcastSource:
    """Wraps a single-sample source to broadcast across any batch size.

    ``n_samples`` always reports ``1``.  ``RoleSourceMap`` excludes
    ``BroadcastSource`` instances from canonical-N resolution so that a
    constant bias vector, for instance, does not force all other sources
    to also have exactly one sample.

    Args:
        inner: An ``ArraySource`` with exactly one sample.  ``get_item``
               always fetches index ``0`` regardless of the requested
               index, and ``get_batch`` stacks that sample ``len(indices)``
               times.

    Example:
        >>> import torch
        >>> from dlkit.engine.data.sources.tensor import TensorSource
        >>> inner = TensorSource(torch.zeros(1, 4))
        >>> src = BroadcastSource(inner)
        >>> src.n_samples
        1
        >>> src.get_item(99).shape
        torch.Size([4])
        >>> src.get_batch([0, 1, 2]).shape
        torch.Size([3, 4])
    """

    def __init__(self, inner: ArraySource) -> None:
        """Initialize with an ``ArraySource`` that holds exactly one sample.

        Args:
            inner: The underlying single-sample source.
        """
        self._inner = inner

    @property
    def n_samples(self) -> int:
        """Always returns ``1`` for broadcast sources.

        Returns:
            Constant integer ``1``.
        """
        return 1

    def get_item(self, idx: int) -> torch.Tensor:
        """Return the single sample, ignoring ``idx``.

        Args:
            idx: Ignored — always fetches sample ``0`` from ``inner``.

        Returns:
            Tensor of shape ``(*sample_shape,)``.
        """
        return self._inner.get_item(0)

    def get_batch(self, indices: list[int]) -> torch.Tensor:
        """Return the single sample stacked ``len(indices)`` times.

        Args:
            indices: Any list of integers; only its length matters.

        Returns:
            Tensor of shape ``(B, *sample_shape)`` where ``B = len(indices)``.
        """
        single = self._inner.get_item(0)
        return torch.stack([single] * len(indices))


__all__ = [
    "BroadcastSource",
    "NamedSources",
]
