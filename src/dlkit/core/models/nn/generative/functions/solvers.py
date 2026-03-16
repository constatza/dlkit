"""Fixed-step ODE solvers for continuous-time flow models.

All solvers are pure functions: no side effects, no state.
The ``integrate`` function composes any solver into a full trajectory.
"""

from collections.abc import Callable

import torch
from torch import Tensor


def euler_step(
    model_fn: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: float,
    dt: float,
) -> Tensor:
    """Single Euler (explicit, first-order) integration step.

    Args:
        model_fn: Vector-field function ``(x, t_tensor) -> dx/dt``.
        x: Current state tensor of shape ``(B, *spatial_dims)``.
        t: Current time as Python float.
        dt: Step size (positive = forward, negative = backward).

    Returns:
        Next state ``x + dt * model_fn(x, t)`` of same shape as ``x``.
    """
    t_tensor = torch.full((x.shape[0],), t, dtype=x.dtype, device=x.device)
    v = model_fn(x, t_tensor)
    return x + dt * v


def heun_step(
    model_fn: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: float,
    dt: float,
) -> Tensor:
    """Single Heun (predictor-corrector, second-order) integration step.

    Args:
        model_fn: Vector-field function ``(x, t_tensor) -> dx/dt``.
        x: Current state tensor of shape ``(B, *spatial_dims)``.
        t: Current time as Python float.
        dt: Step size.

    Returns:
        Corrected next state of same shape as ``x``.
    """
    t_tensor = torch.full((x.shape[0],), t, dtype=x.dtype, device=x.device)
    t_next = t + dt
    t_next_tensor = torch.full((x.shape[0],), t_next, dtype=x.dtype, device=x.device)
    v1 = model_fn(x, t_tensor)
    x_pred = x + dt * v1
    v2 = model_fn(x_pred, t_next_tensor)
    return x + dt * 0.5 * (v1 + v2)


def integrate(
    model_fn: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    t_span: tuple[float, float],
    solver: Callable,
    n_steps: int,
) -> Tensor:
    """Integrate an ODE from ``t_span[0]`` to ``t_span[1]`` using a fixed-step solver.

    Args:
        model_fn: Vector-field function ``(x, t_tensor) -> dx/dt``.
        x0: Initial condition of shape ``(B, *spatial_dims)``.
        t_span: ``(t_start, t_end)`` integration interval.
        solver: Step function matching the ``euler_step`` / ``heun_step`` signature.
        n_steps: Number of uniform steps.

    Returns:
        Final state ``x(t_end)`` of same shape as ``x0``.
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps
    x = x0
    t = t_start
    for _ in range(n_steps):
        x = solver(model_fn, x, t, dt)
        t = t + dt
    return x
