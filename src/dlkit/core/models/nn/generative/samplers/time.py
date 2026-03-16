"""Time samplers for continuous-time flow training."""

import torch
from torch import Tensor


class UniformTimeSampler:
    """Sample time uniformly from [t_min, t_max].

    Implements ``ITimeSampler``.

    Args:
        t_min: Lower bound of time interval (default ``0.0``).
        t_max: Upper bound of time interval (default ``1.0``).
    """

    def __init__(self, t_min: float = 0.0, t_max: float = 1.0) -> None:
        self._t_min = t_min
        self._t_max = t_max

    def __call__(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample batch_size times uniformly from [t_min, t_max].

        Args:
            batch_size: Number of time samples.
            device: Target device.
            dtype: Target dtype.
            generator: Optional RNG for reproducibility.

        Returns:
            Time tensor of shape ``(batch_size,)``.
        """
        return torch.empty(batch_size, device=device, dtype=dtype).uniform_(
            self._t_min, self._t_max
        ) if generator is None else torch.rand(
            batch_size, device=device, dtype=dtype, generator=generator
        ) * (self._t_max - self._t_min) + self._t_min
