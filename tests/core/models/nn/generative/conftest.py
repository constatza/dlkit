"""Shared fixtures for generative module tests.

Provides composable, pure fixtures for tensors and TensorDict batches used
across the generative test suite.  All random state is controlled via explicit
seeds so tests are deterministic and independent.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from torch import Tensor

# ---------------------------------------------------------------------------
# Scalar constants — single source of truth for test dimensions
# ---------------------------------------------------------------------------
_BATCH_SIZE: int = 4
_SPATIAL_DIM: int = 8
_FIXED_SEED: int = 0


@pytest.fixture
def batch_size() -> int:
    """Standard batch size for generative tests.

    Returns:
        Integer batch size of 4.
    """
    return _BATCH_SIZE


@pytest.fixture
def spatial_dim() -> int:
    """Spatial (feature) dimension for generative tests.

    Returns:
        Integer spatial dimension of 8.
    """
    return _SPATIAL_DIM


@pytest.fixture
def device() -> torch.device:
    """Target device for tensor creation.

    Returns:
        CPU torch.device instance.
    """
    return torch.device("cpu")


@pytest.fixture
def dtype() -> torch.dtype:
    """Standard floating-point dtype.

    Returns:
        torch.float32.
    """
    return torch.float32


@pytest.fixture
def x1_tensor(batch_size: int, spatial_dim: int) -> Tensor:
    """Target samples x1 ~ N(0, 1) with fixed seed.

    Args:
        batch_size: Batch size fixture.
        spatial_dim: Spatial dimension fixture.

    Returns:
        Float32 tensor of shape ``(batch_size, spatial_dim)``.
    """
    gen = torch.Generator()
    gen.manual_seed(_FIXED_SEED)
    return torch.randn(batch_size, spatial_dim, generator=gen)


@pytest.fixture
def x0_tensor(batch_size: int, spatial_dim: int) -> Tensor:
    """Source samples x0 — all zeros for easy boundary assertions.

    Args:
        batch_size: Batch size fixture.
        spatial_dim: Spatial dimension fixture.

    Returns:
        Float32 zero tensor of shape ``(batch_size, spatial_dim)``.
    """
    return torch.zeros(batch_size, spatial_dim)


@pytest.fixture
def time_tensor(batch_size: int) -> Tensor:
    """Per-sample time values in [0, 1] with fixed seed.

    Args:
        batch_size: Batch size fixture.

    Returns:
        Float32 tensor of shape ``(batch_size,)`` with values in ``[0, 1]``.
    """
    gen = torch.Generator()
    gen.manual_seed(_FIXED_SEED + 1)
    return torch.rand(batch_size, generator=gen)


@pytest.fixture
def flow_batch(batch_size: int, spatial_dim: int, x1_tensor: Tensor) -> TensorDict:
    """Raw flow matching input batch with x1 in features.

    Contains only ``features["x1"]`` — the supervision builder is expected
    to add ``xt``, ``t``, and ``ut``.

    Args:
        batch_size: Batch size fixture.
        spatial_dim: Spatial dimension fixture.
        x1_tensor: Target sample tensor fixture.

    Returns:
        TensorDict with ``features["x1"]`` and empty ``targets``.
    """
    return TensorDict(
        {
            "features": TensorDict({"x1": x1_tensor}, batch_size=[batch_size]),
            "targets": TensorDict({}, batch_size=[batch_size]),
        },
        batch_size=[batch_size],
    )


@pytest.fixture
def transformed_flow_batch(batch_size: int, spatial_dim: int) -> TensorDict:
    """Pre-transformed batch as produced by FlowMatchingSupervisionBuilder.

    Contains ``features["xt"]``, ``features["t"]``, and ``targets["ut"]``.
    Uses a deterministic generator for reproducibility.

    Args:
        batch_size: Batch size fixture.
        spatial_dim: Spatial dimension fixture.

    Returns:
        TensorDict matching the post-supervision-builder layout.
    """
    gen = torch.Generator()
    gen.manual_seed(_FIXED_SEED + 2)
    x1 = torch.randn(batch_size, spatial_dim, generator=gen)
    x0 = torch.randn(batch_size, spatial_dim, generator=gen)
    t = torch.rand(batch_size, generator=gen)
    t_b = t.unsqueeze(-1)
    xt = (1.0 - t_b) * x0 + t_b * x1
    ut = x1 - x0
    return TensorDict(
        {
            "features": TensorDict({"xt": xt, "t": t}, batch_size=[batch_size]),
            "targets": TensorDict({"ut": ut}, batch_size=[batch_size]),
        },
        batch_size=[batch_size],
    )
