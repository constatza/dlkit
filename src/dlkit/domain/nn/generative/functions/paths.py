"""Interpolation path functions for continuous-time flow models.

All functions are pure: no side effects, no state.
"""

from torch import Tensor

from dlkit.domain.nn.generative.functions.broadcast import broadcast_time


def linear_path(x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    """Compute the linear interpolation path between x0 and x1 at time t.

    Computes ``xt = (1 - t) * x0 + t * x1``, the straight-line interpolation
    used in standard flow matching (Lipman et al., 2022).

    Args:
        x0: Source sample of shape ``(B, *spatial_dims)``.
        x1: Target sample of shape ``(B, *spatial_dims)``.
        t: Time in ``[0, 1]``, shape ``(B,)`` or scalar.

    Returns:
        Interpolated tensor ``xt`` of shape ``(B, *spatial_dims)``.
    """
    t_b = broadcast_time(t, x1)
    return (1.0 - t_b) * x0 + t_b * x1


def noise_schedule_path(x1: Tensor, eps: Tensor, t: Tensor, sigma_min: float = 1e-4) -> Tensor:
    """Compute a noise-schedule interpolation path (Stable Diffusion / EDM style).

    Computes ``xt = (1 - (1 - sigma_min) * t) * eps + t * x1``.

    Args:
        x1: Target sample of shape ``(B, *spatial_dims)``.
        eps: Noise sample of shape ``(B, *spatial_dims)``.
        t: Time in ``[0, 1]``, shape ``(B,)`` or scalar.
        sigma_min: Minimum noise level (default ``1e-4``).

    Returns:
        Noisy interpolated tensor ``xt`` of shape ``(B, *spatial_dims)``.
    """
    t_b = broadcast_time(t, x1)
    return (1.0 - (1.0 - sigma_min) * t_b) * eps + t_b * x1
