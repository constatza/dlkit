"""Tests for generative model samplers.

Covers:
- UniformTimeSampler: output shape, value range, and RNG reproducibility
- GaussianNoiseSampler: output shape, device/dtype matching, and RNG reproducibility
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from dlkit.core.models.nn.generative.samplers.noise import GaussianNoiseSampler
from dlkit.core.models.nn.generative.samplers.time import UniformTimeSampler

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
_BASE_SEED: int = 42
_T_MIN: float = 0.0
_T_MAX: float = 1.0


# ===========================================================================
# UniformTimeSampler
# ===========================================================================


def test_uniform_time_sampler_shape(batch_size: int, device: torch.device) -> None:
    """UniformTimeSampler output is a 1-D tensor of length batch_size.

    Args:
        batch_size: Fixture providing batch size.
        device: Fixture providing target device.
    """
    sampler = UniformTimeSampler()
    result = sampler(batch_size, device=device, dtype=torch.float32)
    assert result.shape == torch.Size([batch_size])


def test_uniform_time_sampler_range(
    batch_size: int, device: torch.device, dtype: torch.dtype
) -> None:
    """UniformTimeSampler values lie within [t_min, t_max].

    Args:
        batch_size: Fixture providing batch size.
        device: Fixture providing target device.
        dtype: Fixture providing float dtype.
    """
    sampler = UniformTimeSampler(t_min=_T_MIN, t_max=_T_MAX)
    # Use many samples to make the range check robust.
    result = sampler(batch_size * 64, device=device, dtype=dtype)
    assert result.min() >= _T_MIN
    assert result.max() <= _T_MAX


def test_uniform_time_sampler_dtype_respected(batch_size: int) -> None:
    """UniformTimeSampler output dtype matches requested dtype.

    Args:
        batch_size: Fixture providing batch size.
    """
    sampler = UniformTimeSampler()
    result = sampler(batch_size, device=torch.device("cpu"), dtype=torch.float64)
    assert result.dtype == torch.float64


def test_uniform_time_sampler_with_generator(
    batch_size: int, device: torch.device, dtype: torch.dtype
) -> None:
    """Seeded generator makes UniformTimeSampler reproducible.

    Two independent calls with the same seed must return identical tensors.

    Args:
        batch_size: Fixture providing batch size.
        device: Fixture providing target device.
        dtype: Fixture providing float dtype.
    """
    sampler = UniformTimeSampler()

    gen1 = torch.Generator(device=device)
    gen1.manual_seed(_BASE_SEED)
    t1 = sampler(batch_size, device=device, dtype=dtype, generator=gen1)

    gen2 = torch.Generator(device=device)
    gen2.manual_seed(_BASE_SEED)
    t2 = sampler(batch_size, device=device, dtype=dtype, generator=gen2)

    assert torch.equal(t1, t2)


def test_uniform_time_sampler_custom_range(batch_size: int) -> None:
    """UniformTimeSampler with custom [t_min, t_max] respects boundaries.

    Args:
        batch_size: Fixture providing batch size.
    """
    t_min, t_max = 0.1, 0.9
    sampler = UniformTimeSampler(t_min=t_min, t_max=t_max)
    gen = torch.Generator()
    gen.manual_seed(_BASE_SEED)
    result = sampler(batch_size * 64, device=torch.device("cpu"), dtype=torch.float32, generator=gen)
    assert result.min() >= t_min
    assert result.max() <= t_max


# ===========================================================================
# GaussianNoiseSampler
# ===========================================================================


def test_gaussian_noise_sampler_shape(x1_tensor: Tensor) -> None:
    """GaussianNoiseSampler output shape matches reference tensor shape.

    Args:
        x1_tensor: Reference tensor fixture.
    """
    sampler = GaussianNoiseSampler()
    result = sampler(x1_tensor)
    assert result.shape == x1_tensor.shape


def test_gaussian_noise_sampler_device_dtype(x1_tensor: Tensor) -> None:
    """GaussianNoiseSampler output device and dtype match the reference tensor.

    Args:
        x1_tensor: Reference tensor fixture.
    """
    sampler = GaussianNoiseSampler()
    result = sampler(x1_tensor)
    assert result.device == x1_tensor.device
    assert result.dtype == x1_tensor.dtype


def test_gaussian_noise_sampler_3d_shape(batch_size: int, spatial_dim: int) -> None:
    """GaussianNoiseSampler handles 3-D reference tensors (B, C, L).

    Args:
        batch_size: Fixture providing batch size.
        spatial_dim: Fixture providing spatial dimension (used as both C and L).
    """
    ref = torch.zeros(batch_size, 2, spatial_dim)
    sampler = GaussianNoiseSampler()
    result = sampler(ref)
    assert result.shape == ref.shape


def test_gaussian_noise_sampler_with_generator(x1_tensor: Tensor) -> None:
    """Seeded generator makes GaussianNoiseSampler reproducible.

    Two independent calls with the same seed must return identical tensors.

    Args:
        x1_tensor: Reference tensor fixture.
    """
    sampler = GaussianNoiseSampler()

    gen1 = torch.Generator()
    gen1.manual_seed(_BASE_SEED)
    noise1 = sampler(x1_tensor, gen1)

    gen2 = torch.Generator()
    gen2.manual_seed(_BASE_SEED)
    noise2 = sampler(x1_tensor, gen2)

    assert torch.equal(noise1, noise2)


def test_gaussian_noise_sampler_different_seeds_differ(x1_tensor: Tensor) -> None:
    """Different seeds produce different noise samples (probabilistic sanity check).

    Args:
        x1_tensor: Reference tensor fixture.
    """
    sampler = GaussianNoiseSampler()

    gen1 = torch.Generator()
    gen1.manual_seed(1)
    noise1 = sampler(x1_tensor, gen1)

    gen2 = torch.Generator()
    gen2.manual_seed(2)
    noise2 = sampler(x1_tensor, gen2)

    assert not torch.equal(noise1, noise2)
