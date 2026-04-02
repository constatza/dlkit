"""Noise samplers for generative model initial conditions."""

import torch
from torch import Tensor


class GaussianNoiseSampler:
    """Sample standard Gaussian noise matching shape/device/dtype of a reference tensor.

    Implements ``INoiseSampler``.
    """

    def __call__(
        self,
        ref: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample standard Gaussian noise with same shape, device, dtype as ref.

        Args:
            ref: Reference tensor to infer shape/device/dtype from.
            generator: Optional RNG for reproducibility.

        Returns:
            Gaussian noise tensor of same shape as ``ref``.
        """
        return torch.randn(ref.shape, dtype=ref.dtype, device=ref.device, generator=generator)
