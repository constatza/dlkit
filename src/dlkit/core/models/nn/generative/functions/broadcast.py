"""Broadcast utilities for continuous-time flow models."""

from torch import Tensor


def broadcast_time(t: Tensor, ref: Tensor) -> Tensor:
    """Broadcast time tensor to match the spatial dimensions of ref.

    Expands a scalar or per-sample time tensor ``t`` of shape ``(B,)`` or ``()``
    to ``(B, 1, 1, ...)`` so element-wise ops with ``ref`` broadcast correctly.

    Args:
        t: Time tensor of shape ``(B,)`` or scalar ``()``.
        ref: Reference spatial tensor of shape ``(B, *spatial_dims)``.

    Returns:
        Tensor of shape ``(B, 1, ..., 1)`` with ``ref.ndim - 1`` trailing ones,
        same device and dtype as ``t``.

    Example:
        >>> t = torch.rand(4)          # (B,)
        >>> x = torch.rand(4, 3, 32)  # (B, C, L)
        >>> broadcast_time(t, x).shape
        torch.Size([4, 1, 1])
    """
    extra_dims = ref.ndim - 1
    return t.view(-1, *([1] * extra_dims))
