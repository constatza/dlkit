"""Pure functions for Lightning wrapper computations.

These functions are pure (no side effects, no self) and easily testable.
They handle core processing operations like loss computation and transform application.

Design Pattern: Functional Programming
- Pure functions with no side effects
- Input/output contracts clearly defined via type hints
- Easy to test without mocking or fixtures
- Composable and reusable across wrappers
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


def compute_loss(
    predictions: Tensor,
    targets: tuple[Tensor, ...],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Compute loss from predictions and positional targets.

    Pure function — no side effects, no state mutation.

    Args:
        predictions: Model output tensor.
        targets: Tuple of target tensors (uses first target).
        loss_fn: Loss function callable with signature (predictions, target) -> loss.

    Returns:
        Scalar loss tensor.

    Example:
        >>> predictions = torch.randn(32, 10)
        >>> targets = (torch.randint(0, 10, (32,)),)
        >>> loss_fn = torch.nn.CrossEntropyLoss()
        >>> loss = compute_loss(predictions, targets, loss_fn)
    """
    return loss_fn(predictions, targets[0].to(predictions.dtype))


def apply_transforms(
    tensors: tuple[Tensor, ...],
    transforms: tuple[Any | None, ...],
) -> tuple[Tensor, ...]:
    """Apply per-position transforms. Pure function.

    Applies a transform to each position in the tensors tuple. Positions with
    None transform are passed through unchanged (identity).

    Args:
        tensors: Input tensors to transform (same length as transforms).
        transforms: Per-position transform callables (None = identity).

    Returns:
        Transformed tensors tuple (same length as inputs).

    Example:
        >>> scaler = MinMaxScaler(dim=0)
        >>> scaler.fit(train_features)
        >>> features = torch.randn(32, 64)
        >>> targets = torch.randn(32, 1)
        >>> transformed = apply_transforms(
        ...     (features, targets),
        ...     (scaler, None),  # Scale features, keep targets
        ... )
    """
    return tuple(t(x) if t is not None else x for x, t in zip(tensors, transforms))


def apply_chain(x: Tensor, chain: "torch.nn.Module") -> Tensor:
    """Apply a transform chain forward pass. Pure function.

    Args:
        x: Input tensor.
        chain: Transform chain (TransformChain or nn.Identity).

    Returns:
        Transformed tensor.
    """
    return chain(x)


def apply_inverse_chain(x: Tensor, chain: "torch.nn.Module") -> Tensor:
    """Apply inverse transform if chain supports it, otherwise return x unchanged. Pure function.

    Args:
        x: Input tensor (model output or prediction).
        chain: Transform chain to potentially invert.

    Returns:
        Inverse-transformed tensor, or x unchanged if chain is not invertible.
    """
    from dlkit.core.training.transforms.base import InvertibleTransform

    if isinstance(chain, InvertibleTransform):
        return chain.inverse_transform(x)
    return x
