"""Tests for pure generative function utilities.

Covers:
- broadcast_time: shape broadcasting for 1-D and 2-D spatial refs
- linear_path: boundary conditions (t=0 → x0, t=1 → x1) and shape preservation
- noise_schedule_path: boundary conditions and shape preservation
- displacement_target: value identity and shape
- euler_step: constant-velocity linear update
- heun_step: zero-model no-change invariant
- integrate: output shape and step count consumption
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from dlkit.core.models.nn.generative.functions.broadcast import broadcast_time
from dlkit.core.models.nn.generative.functions.paths import linear_path, noise_schedule_path
from dlkit.core.models.nn.generative.functions.solvers import euler_step, heun_step, integrate
from dlkit.core.models.nn.generative.functions.targets import displacement_target

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
_T_ZERO: float = 0.0
_T_ONE: float = 1.0
_SIGMA_MIN: float = 1e-4
_ABS_TOL: float = 1e-5


# ===========================================================================
# broadcast_time
# ===========================================================================


def test_broadcast_time_1d_spatial(batch_size: int) -> None:
    """broadcast_time with 2-D ref (B, D) → result shape (B, 1).

    Args:
        batch_size: Fixture providing batch size.
    """
    spatial_dim = 8
    t = torch.rand(batch_size)
    ref = torch.randn(batch_size, spatial_dim)
    result = broadcast_time(t, ref)
    assert result.shape == torch.Size([batch_size, 1])


def test_broadcast_time_2d_spatial(batch_size: int) -> None:
    """broadcast_time with 3-D ref (B, C, L) → result shape (B, 1, 1).

    Args:
        batch_size: Fixture providing batch size.
    """
    channels, length = 3, 16
    t = torch.rand(batch_size)
    ref = torch.randn(batch_size, channels, length)
    result = broadcast_time(t, ref)
    assert result.shape == torch.Size([batch_size, 1, 1])


def test_broadcast_time_scalar() -> None:
    """Scalar time tensor broadcasts to single-element leading dimension."""
    t = torch.tensor(0.5)
    ref = torch.randn(1, 4)
    result = broadcast_time(t, ref)
    # scalar viewed as (1,) → (1, 1) after one trailing expansion
    assert result.ndim == ref.ndim


def test_broadcast_time_dtype_preserved(batch_size: int) -> None:
    """broadcast_time output dtype matches input time tensor dtype.

    Args:
        batch_size: Fixture providing batch size.
    """
    t = torch.rand(batch_size, dtype=torch.float64)
    ref = torch.randn(batch_size, 4, dtype=torch.float32)
    result = broadcast_time(t, ref)
    assert result.dtype == torch.float64


# ===========================================================================
# linear_path
# ===========================================================================


def test_linear_path_at_t0_equals_x0(x0_tensor: Tensor, x1_tensor: Tensor, batch_size: int) -> None:
    """linear_path at t=0 returns x0 exactly.

    Args:
        x0_tensor: Source sample fixture (all zeros).
        x1_tensor: Target sample fixture.
        batch_size: Fixture providing batch size.
    """
    t_zero = torch.zeros(batch_size)
    result = linear_path(x0_tensor, x1_tensor, t_zero)
    assert torch.allclose(result, x0_tensor, atol=_ABS_TOL)


def test_linear_path_at_t1_equals_x1(x0_tensor: Tensor, x1_tensor: Tensor, batch_size: int) -> None:
    """linear_path at t=1 returns x1 exactly.

    Args:
        x0_tensor: Source sample fixture.
        x1_tensor: Target sample fixture.
        batch_size: Fixture providing batch size.
    """
    t_one = torch.ones(batch_size)
    result = linear_path(x0_tensor, x1_tensor, t_one)
    assert torch.allclose(result, x1_tensor, atol=_ABS_TOL)


def test_linear_path_shape(x0_tensor: Tensor, x1_tensor: Tensor, time_tensor: Tensor) -> None:
    """linear_path output shape equals input spatial shape.

    Args:
        x0_tensor: Source sample fixture.
        x1_tensor: Target sample fixture.
        time_tensor: Per-sample time fixture.
    """
    result = linear_path(x0_tensor, x1_tensor, time_tensor)
    assert result.shape == x1_tensor.shape


def test_linear_path_midpoint(x0_tensor: Tensor, x1_tensor: Tensor, batch_size: int) -> None:
    """linear_path at t=0.5 returns the midpoint of x0 and x1.

    Args:
        x0_tensor: Source sample fixture.
        x1_tensor: Target sample fixture.
        batch_size: Fixture providing batch size.
    """
    t_half = torch.full((batch_size,), 0.5)
    result = linear_path(x0_tensor, x1_tensor, t_half)
    expected = 0.5 * x0_tensor + 0.5 * x1_tensor
    assert torch.allclose(result, expected, atol=_ABS_TOL)


# ===========================================================================
# noise_schedule_path
# ===========================================================================


def test_noise_schedule_path_at_t0(x1_tensor: Tensor, x0_tensor: Tensor, batch_size: int) -> None:
    """noise_schedule_path at t=0 returns eps (the noise, here x0_tensor).

    At t=0: xt = (1 - 0) * eps + 0 * x1 = eps.

    Args:
        x1_tensor: Target sample fixture.
        x0_tensor: Noise sample fixture (used as eps).
        batch_size: Fixture providing batch size.
    """
    t_zero = torch.zeros(batch_size)
    result = noise_schedule_path(x1_tensor, x0_tensor, t_zero, sigma_min=_SIGMA_MIN)
    assert torch.allclose(result, x0_tensor, atol=_ABS_TOL)


def test_noise_schedule_path_at_t1(x1_tensor: Tensor, x0_tensor: Tensor, batch_size: int) -> None:
    """noise_schedule_path at t=1 equals sigma_min * eps + x1.

    At t=1: xt = (1 - (1 - sigma_min)) * eps + 1 * x1 = sigma_min * eps + x1.

    Args:
        x1_tensor: Target sample fixture.
        x0_tensor: Noise sample fixture (used as eps).
        batch_size: Fixture providing batch size.
    """
    t_one = torch.ones(batch_size)
    result = noise_schedule_path(x1_tensor, x0_tensor, t_one, sigma_min=_SIGMA_MIN)
    expected = _SIGMA_MIN * x0_tensor + x1_tensor
    assert torch.allclose(result, expected, atol=_ABS_TOL)


def test_noise_schedule_path_shape(
    x1_tensor: Tensor, x0_tensor: Tensor, time_tensor: Tensor
) -> None:
    """noise_schedule_path output shape equals x1 shape.

    Args:
        x1_tensor: Target sample fixture.
        x0_tensor: Noise/eps sample fixture.
        time_tensor: Per-sample time fixture.
    """
    result = noise_schedule_path(x1_tensor, x0_tensor, time_tensor)
    assert result.shape == x1_tensor.shape


# ===========================================================================
# displacement_target
# ===========================================================================


def test_displacement_target_shape(x0_tensor: Tensor, x1_tensor: Tensor) -> None:
    """displacement_target output shape equals input spatial shape.

    Args:
        x0_tensor: Source sample fixture.
        x1_tensor: Target sample fixture.
    """
    result = displacement_target(x0_tensor, x1_tensor)
    assert result.shape == x1_tensor.shape


def test_displacement_target_value(x0_tensor: Tensor, x1_tensor: Tensor) -> None:
    """displacement_target equals x1 - x0.

    Args:
        x0_tensor: Source sample fixture.
        x1_tensor: Target sample fixture.
    """
    result = displacement_target(x0_tensor, x1_tensor)
    expected = x1_tensor - x0_tensor
    assert torch.allclose(result, expected, atol=_ABS_TOL)


def test_displacement_target_x0_zero_equals_x1(x0_tensor: Tensor, x1_tensor: Tensor) -> None:
    """When x0 is all-zeros, displacement_target returns x1 exactly.

    Args:
        x0_tensor: Source sample fixture (all zeros).
        x1_tensor: Target sample fixture.
    """
    result = displacement_target(x0_tensor, x1_tensor)
    assert torch.allclose(result, x1_tensor, atol=_ABS_TOL)


# ===========================================================================
# euler_step
# ===========================================================================


def test_euler_step_constant_model(batch_size: int, spatial_dim: int) -> None:
    """Constant velocity model → euler_step produces linear update x + dt * v.

    Args:
        batch_size: Fixture providing batch size.
        spatial_dim: Fixture providing spatial dimension.
    """
    velocity = torch.ones(batch_size, spatial_dim) * 2.0
    x = torch.zeros(batch_size, spatial_dim)
    dt = 0.1

    def constant_model(x_in: Tensor, t_in: Tensor) -> Tensor:
        """Always returns the pre-defined constant velocity."""
        return velocity

    result = euler_step(constant_model, x, t=0.0, dt=dt)
    expected = x + dt * velocity
    assert torch.allclose(result, expected, atol=_ABS_TOL)


def test_euler_step_shape_preserved(batch_size: int, spatial_dim: int) -> None:
    """euler_step output shape matches input state shape.

    Args:
        batch_size: Fixture providing batch size.
        spatial_dim: Fixture providing spatial dimension.
    """
    x = torch.randn(batch_size, spatial_dim)

    def zero_model(x_in: Tensor, t_in: Tensor) -> Tensor:
        """Returns zero velocity."""
        return torch.zeros_like(x_in)

    result = euler_step(zero_model, x, t=0.5, dt=0.1)
    assert result.shape == x.shape


def test_euler_step_t_tensor_dtype(batch_size: int, spatial_dim: int) -> None:
    """euler_step passes a float32 t tensor matching x dtype to model_fn.

    Args:
        batch_size: Fixture providing batch size.
        spatial_dim: Fixture providing spatial dimension.
    """
    received_t: list[Tensor] = []
    x = torch.zeros(batch_size, spatial_dim, dtype=torch.float32)

    def capture_t(x_in: Tensor, t_in: Tensor) -> Tensor:
        """Captures the time tensor passed by euler_step."""
        received_t.append(t_in)
        return torch.zeros_like(x_in)

    euler_step(capture_t, x, t=0.3, dt=0.1)
    assert received_t[0].dtype == torch.float32
    assert received_t[0].shape == (batch_size,)


# ===========================================================================
# heun_step
# ===========================================================================


def test_heun_step_zero_model(batch_size: int, spatial_dim: int) -> None:
    """Zero velocity model → heun_step leaves x unchanged.

    Args:
        batch_size: Fixture providing batch size.
        spatial_dim: Fixture providing spatial dimension.
    """
    x = torch.randn(batch_size, spatial_dim)

    def zero_model(x_in: Tensor, t_in: Tensor) -> Tensor:
        """Returns zero velocity."""
        return torch.zeros_like(x_in)

    result = heun_step(zero_model, x, t=0.0, dt=0.1)
    assert torch.allclose(result, x, atol=_ABS_TOL)


def test_heun_step_shape_preserved(batch_size: int, spatial_dim: int) -> None:
    """heun_step output shape matches input state shape.

    Args:
        batch_size: Fixture providing batch size.
        spatial_dim: Fixture providing spatial dimension.
    """
    x = torch.randn(batch_size, spatial_dim)

    def constant_model(x_in: Tensor, t_in: Tensor) -> Tensor:
        """Returns constant velocity of 1.0."""
        return torch.ones_like(x_in)

    result = heun_step(constant_model, x, t=0.0, dt=0.05)
    assert result.shape == x.shape


def test_heun_step_constant_model_equals_euler(batch_size: int, spatial_dim: int) -> None:
    """For a constant-velocity model, Heun and Euler give identical results.

    When the velocity field is constant (no dependence on x or t), the
    Heun corrector equals the Euler predictor because v1 == v2.

    Args:
        batch_size: Fixture providing batch size.
        spatial_dim: Fixture providing spatial dimension.
    """
    x = torch.randn(batch_size, spatial_dim)
    v = torch.ones_like(x) * 3.0
    dt = 0.05

    def constant_model(x_in: Tensor, t_in: Tensor) -> Tensor:
        """Returns the pre-defined constant velocity."""
        return v

    euler_result = euler_step(constant_model, x, t=0.0, dt=dt)
    heun_result = heun_step(constant_model, x, t=0.0, dt=dt)
    assert torch.allclose(euler_result, heun_result, atol=_ABS_TOL)


# ===========================================================================
# integrate
# ===========================================================================


def test_integrate_shape_preserved(batch_size: int, spatial_dim: int) -> None:
    """integrate output shape matches x0 shape.

    Args:
        batch_size: Fixture providing batch size.
        spatial_dim: Fixture providing spatial dimension.
    """
    x0 = torch.randn(batch_size, spatial_dim)

    def zero_model(x_in: Tensor, t_in: Tensor) -> Tensor:
        """Returns zero velocity."""
        return torch.zeros_like(x_in)

    result = integrate(zero_model, x0, t_span=(0.0, 1.0), solver=euler_step, n_steps=5)
    assert result.shape == x0.shape


def test_integrate_zero_model_identity(batch_size: int, spatial_dim: int) -> None:
    """Zero velocity model → integrate returns x0 unchanged.

    Args:
        batch_size: Fixture providing batch size.
        spatial_dim: Fixture providing spatial dimension.
    """
    x0 = torch.randn(batch_size, spatial_dim)

    def zero_model(x_in: Tensor, t_in: Tensor) -> Tensor:
        """Returns zero velocity."""
        return torch.zeros_like(x_in)

    result = integrate(zero_model, x0, t_span=(0.0, 1.0), solver=euler_step, n_steps=10)
    assert torch.allclose(result, x0, atol=_ABS_TOL)


def test_integrate_n_steps_consumed(batch_size: int, spatial_dim: int) -> None:
    """integrate calls the solver exactly n_steps times.

    Args:
        batch_size: Fixture providing batch size.
        spatial_dim: Fixture providing spatial dimension.
    """
    call_count: list[int] = [0]

    def counting_solver(
        model_fn: Callable,
        x: Tensor,
        t: float,
        dt: float,
    ) -> Tensor:
        """Euler step that increments a counter on each call."""
        call_count[0] += 1
        return euler_step(model_fn, x, t, dt)

    def zero_model(x_in: Tensor, t_in: Tensor) -> Tensor:
        """Returns zero velocity."""
        return torch.zeros_like(x_in)

    n_steps = 7
    x0 = torch.randn(batch_size, spatial_dim)
    integrate(zero_model, x0, t_span=(0.0, 1.0), solver=counting_solver, n_steps=n_steps)
    assert call_count[0] == n_steps


def test_integrate_heun_same_shape(batch_size: int, spatial_dim: int) -> None:
    """integrate with heun_step produces same shape as with euler_step.

    Args:
        batch_size: Fixture providing batch size.
        spatial_dim: Fixture providing spatial dimension.
    """
    x0 = torch.randn(batch_size, spatial_dim)

    def zero_model(x_in: Tensor, t_in: Tensor) -> Tensor:
        """Returns zero velocity."""
        return torch.zeros_like(x_in)

    result = integrate(zero_model, x0, t_span=(0.0, 1.0), solver=heun_step, n_steps=5)
    assert result.shape == x0.shape
