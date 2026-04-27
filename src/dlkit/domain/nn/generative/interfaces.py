"""Narrow single-method protocols for continuous-time flow model components.

Each protocol has exactly one method — ISP compliance.
All are ``@runtime_checkable`` so isinstance() checks work at runtime.
"""

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class ITimeSampler(Protocol):
    """Sample time values in [0, 1] for stochastic training supervision."""

    def __call__(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample a batch of times.

        Args:
            batch_size: Number of samples.
            device: Target device.
            dtype: Target dtype.
            generator: Optional RNG for reproducibility.

        Returns:
            Time tensor of shape ``(batch_size,)`` in ``[0, 1]``.
        """
        ...


@runtime_checkable
class INoiseSampler(Protocol):
    """Sample noise / initial condition for ODE integration."""

    def __call__(
        self,
        ref: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample noise matching shape, device, and dtype of ``ref``.

        Args:
            ref: Reference tensor to infer shape/device/dtype from.
            generator: Optional RNG for reproducibility.

        Returns:
            Noise tensor of same shape as ``ref``.
        """
        ...


@runtime_checkable
class IInterpolationPath(Protocol):
    """Compute an interpolated sample at time t given source x0 and target x1."""

    def __call__(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Interpolate between x0 and x1 at time t.

        Args:
            x0: Source sample ``(B, *spatial_dims)``.
            x1: Target sample ``(B, *spatial_dims)``.
            t: Time tensor ``(B,)`` in ``[0, 1]``.

        Returns:
            Interpolated tensor of same shape as ``x1``.
        """
        ...


@runtime_checkable
class IVelocityTarget(Protocol):
    """Compute the velocity (supervision) target given source and target."""

    def __call__(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Compute velocity target.

        Args:
            x0: Source sample ``(B, *spatial_dims)``.
            x1: Target sample ``(B, *spatial_dims)``.

        Returns:
            Velocity target of same shape as ``x1``.
        """
        ...


@runtime_checkable
class IModelAdapter(Protocol):
    """Adapt a model call to the ``(x, t, context)`` signature expected by ODE solvers."""

    def __call__(
        self,
        model: torch.nn.Module,
        x: Tensor,
        t: Tensor,
        context: dict | None = None,
    ) -> Tensor:
        """Invoke model and return velocity estimate.

        Args:
            model: The neural network to call.
            x: Current state tensor ``(B, *spatial_dims)``.
            t: Time tensor ``(B,)``.
            context: Optional conditioning context.

        Returns:
            Velocity estimate of same shape as ``x``.
        """
        ...


@runtime_checkable
class IFixedStepSolver(Protocol):
    """Single fixed-step ODE integration step."""

    def __call__(
        self,
        model_fn: IModelFn,
        x: Tensor,
        t: float,
        dt: float,
    ) -> Tensor:
        """Advance state one step.

        Args:
            model_fn: ``(x, t_tensor) -> velocity`` callable.
            x: Current state.
            t: Current time (Python float).
            dt: Step size.

        Returns:
            Next state of same shape as ``x``.
        """
        ...


@runtime_checkable
class IModelFn(Protocol):
    """Model function consumed by ODE solvers: ``(x, t) -> velocity``."""

    def __call__(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute velocity at (x, t).

        Args:
            x: State tensor.
            t: Time tensor.

        Returns:
            Velocity tensor of same shape as ``x``.
        """
        ...
