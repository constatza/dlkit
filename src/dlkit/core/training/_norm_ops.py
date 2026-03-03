"""Internal functional primitives for norm-based losses and metrics.

This module centralizes pure composition helpers shared by:
    - dlkit.core.training.metrics.functional
    - dlkit.core.training.functional

It is intentionally internal and not part of the public API.
"""

from collections.abc import Callable

from torch import Tensor

ErrorFn = Callable[[Tensor, Tensor], Tensor]
NormFn = Callable[[Tensor], Tensor]
DivideFn = Callable[[Tensor, Tensor, float], Tensor]


def compute_error_norms(
    preds: Tensor,
    target: Tensor,
    *,
    error_fn: ErrorFn,
    norm_fn: NormFn,
) -> Tensor:
    """Compute norms of prediction errors.

    Returns per-sample or per-element norms depending on ``norm_fn``.
    """
    error_vectors = error_fn(preds, target)
    return norm_fn(error_vectors)


def compute_relative_norms(
    preds: Tensor,
    target: Tensor,
    *,
    error_fn: ErrorFn,
    norm_fn: NormFn,
    divide_fn: DivideFn,
    eps: float,
) -> Tensor:
    """Compute relative norms: ||pred - target|| / ||target||."""
    error_norms = compute_error_norms(
        preds,
        target,
        error_fn=error_fn,
        norm_fn=norm_fn,
    )
    target_norms = norm_fn(target)
    return divide_fn(error_norms, target_norms, eps=eps)


def compute_state_mean(sum_value: Tensor, total: int | Tensor) -> Tensor:
    """Compute mean from accumulated sum and count.

    Behavior intentionally mirrors raw division semantics used by metric
    wrappers, including ``total == 0`` behavior.
    """
    return sum_value / total
