"""Velocity target functions for continuous-time flow models.

All functions are pure: no side effects, no state.
"""

from torch import Tensor


def displacement_target(x0: Tensor, x1: Tensor) -> Tensor:
    """Compute the displacement velocity target for linear flow matching.

    Returns the constant velocity ``ut = x1 - x0`` that corresponds to
    straight-line paths in standard flow matching.

    Args:
        x0: Source sample of shape ``(B, *spatial_dims)``.
        x1: Target sample of shape ``(B, *spatial_dims)``.

    Returns:
        Velocity target ``ut`` of shape ``(B, *spatial_dims)``.
    """
    return x1 - x0
