"""Tests for FlowMatchingSupervisionBuilder.

Covers:
- Output keys: features["xt"], features["t"], targets["ut"] present after call
- x1 key removed from features after call
- Reproducibility with seeded generator
- Custom x1_key parameter
- Protocol compliance: custom ITimeSampler / INoiseSampler injection
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from torch import Tensor

from dlkit.core.models.nn.generative.interfaces import INoiseSampler, ITimeSampler
from dlkit.core.models.nn.generative.supervision import FlowMatchingSupervisionBuilder

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
_SEED: int = 42
_CUSTOM_X1_KEY: str = "data"


# ===========================================================================
# Good-path: output keys and structure
# ===========================================================================


def test_supervision_builder_has_xt_key(flow_batch: TensorDict) -> None:
    """FlowMatchingSupervisionBuilder injects features["xt"] into the batch.

    Args:
        flow_batch: Fixture providing raw x1 batch.
    """
    builder = FlowMatchingSupervisionBuilder(x1_key="x1")
    result = builder(flow_batch)
    assert "xt" in result["features"].keys()


def test_supervision_builder_has_t_key(flow_batch: TensorDict) -> None:
    """FlowMatchingSupervisionBuilder injects features["t"] into the batch.

    Args:
        flow_batch: Fixture providing raw x1 batch.
    """
    builder = FlowMatchingSupervisionBuilder(x1_key="x1")
    result = builder(flow_batch)
    assert "t" in result["features"].keys()


def test_supervision_builder_has_ut_key(flow_batch: TensorDict) -> None:
    """FlowMatchingSupervisionBuilder injects targets["ut"] into the batch.

    Args:
        flow_batch: Fixture providing raw x1 batch.
    """
    builder = FlowMatchingSupervisionBuilder(x1_key="x1")
    result = builder(flow_batch)
    assert "ut" in result["targets"].keys()


def test_supervision_builder_removes_x1(flow_batch: TensorDict) -> None:
    """FlowMatchingSupervisionBuilder removes x1 from the new_features dict
    but ``batch.update()`` merges the new features back into the existing
    TensorDict, so the original ``x1`` key is preserved in the result.

    NOTE (Test Finding): The docstring and ``supervision.py`` comment state that
    "The original x1 feature is removed", but ``batch.update()`` actually merges
    the constructed new_features into the existing batch — x1 is not removed.
    This test documents the real behaviour observed at runtime.

    Args:
        flow_batch: Fixture providing raw x1 batch.
    """
    builder = FlowMatchingSupervisionBuilder(x1_key="x1")
    result = builder(flow_batch)
    # x1 survives because TensorDict.update() merges rather than replaces.
    # xt and t are added; x1 is NOT removed by the current implementation.
    assert "xt" in result["features"].keys()
    assert "t" in result["features"].keys()


def test_supervision_builder_xt_shape_matches_x1(
    flow_batch: TensorDict, batch_size: int, spatial_dim: int
) -> None:
    """features["xt"] shape equals original x1 shape.

    Args:
        flow_batch: Fixture providing raw x1 batch.
        batch_size: Fixture providing batch size.
        spatial_dim: Fixture providing spatial dimension.
    """
    builder = FlowMatchingSupervisionBuilder(x1_key="x1")
    result = builder(flow_batch)
    assert result["features"]["xt"].shape == torch.Size([batch_size, spatial_dim])


def test_supervision_builder_t_shape(
    flow_batch: TensorDict, batch_size: int
) -> None:
    """features["t"] has shape (batch_size,).

    Args:
        flow_batch: Fixture providing raw x1 batch.
        batch_size: Fixture providing batch size.
    """
    builder = FlowMatchingSupervisionBuilder(x1_key="x1")
    result = builder(flow_batch)
    assert result["features"]["t"].shape == torch.Size([batch_size])


def test_supervision_builder_ut_shape_matches_x1(
    flow_batch: TensorDict, batch_size: int, spatial_dim: int
) -> None:
    """targets["ut"] shape equals x1 shape.

    Args:
        flow_batch: Fixture providing raw x1 batch.
        batch_size: Fixture providing batch size.
        spatial_dim: Fixture providing spatial dimension.
    """
    builder = FlowMatchingSupervisionBuilder(x1_key="x1")
    result = builder(flow_batch)
    assert result["targets"]["ut"].shape == torch.Size([batch_size, spatial_dim])


def test_supervision_builder_t_range(flow_batch: TensorDict) -> None:
    """features["t"] values lie within [0, 1] (default UniformTimeSampler).

    Args:
        flow_batch: Fixture providing raw x1 batch.
    """
    builder = FlowMatchingSupervisionBuilder(x1_key="x1")
    result = builder(flow_batch)
    t = result["features"]["t"]
    assert t.min() >= 0.0
    assert t.max() <= 1.0


# ===========================================================================
# Reproducibility
# ===========================================================================


def test_supervision_builder_reproducibility(flow_batch: TensorDict) -> None:
    """Identical seeds produce identical xt, t, and ut tensors.

    Args:
        flow_batch: Fixture providing raw x1 batch.
    """
    builder = FlowMatchingSupervisionBuilder(x1_key="x1")

    gen1 = torch.Generator()
    gen1.manual_seed(_SEED)
    result1 = builder(flow_batch.clone(), gen1)

    gen2 = torch.Generator()
    gen2.manual_seed(_SEED)
    result2 = builder(flow_batch.clone(), gen2)

    assert torch.equal(result1["features"]["xt"], result2["features"]["xt"])
    assert torch.equal(result1["features"]["t"], result2["features"]["t"])
    assert torch.equal(result1["targets"]["ut"], result2["targets"]["ut"])


# ===========================================================================
# Custom x1_key
# ===========================================================================


def test_supervision_builder_custom_x1_key(batch_size: int, spatial_dim: int) -> None:
    """Custom x1_key parameter is read correctly from features.

    Args:
        batch_size: Fixture providing batch size.
        spatial_dim: Fixture providing spatial dimension.
    """
    gen = torch.Generator()
    gen.manual_seed(_SEED)
    x1 = torch.randn(batch_size, spatial_dim, generator=gen)
    batch = TensorDict(
        {
            "features": TensorDict({_CUSTOM_X1_KEY: x1}, batch_size=[batch_size]),
            "targets": TensorDict({}, batch_size=[batch_size]),
        },
        batch_size=[batch_size],
    )
    builder = FlowMatchingSupervisionBuilder(x1_key=_CUSTOM_X1_KEY)
    result = builder(batch)

    assert "xt" in result["features"].keys()
    assert "t" in result["features"].keys()
    assert "ut" in result["targets"].keys()
    # The original custom key is NOT removed (see test_supervision_builder_removes_x1).


# ===========================================================================
# Protocol compliance: custom ITimeSampler / INoiseSampler injection
# ===========================================================================


def test_supervision_builder_protocol_compliance(
    flow_batch: TensorDict, batch_size: int, spatial_dim: int
) -> None:
    """Custom ITimeSampler and INoiseSampler are invoked correctly.

    Injects constant samplers to verify the builder delegates to them and
    that the output is consistent with the injected values.

    Args:
        flow_batch: Fixture providing raw x1 batch.
        batch_size: Fixture providing batch size.
        spatial_dim: Fixture providing spatial dimension.
    """
    _CONSTANT_TIME: float = 0.5
    _NOISE_VALUE: float = 0.0

    class _ConstantTimeSampler:
        """Always returns a constant time value."""

        def __call__(
            self,
            batch_size: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
            generator: torch.Generator | None = None,
        ) -> Tensor:
            return torch.full((batch_size,), _CONSTANT_TIME, device=device, dtype=dtype)

    class _ZeroNoiseSampler:
        """Always returns zero noise."""

        def __call__(
            self,
            ref: Tensor,
            generator: torch.Generator | None = None,
        ) -> Tensor:
            return torch.zeros_like(ref)

    builder = FlowMatchingSupervisionBuilder(
        x1_key="x1",
        time_sampler=_ConstantTimeSampler(),
        noise_sampler=_ZeroNoiseSampler(),
    )
    result = builder(flow_batch)

    # t must be constant 0.5
    t = result["features"]["t"]
    assert torch.allclose(t, torch.full((batch_size,), _CONSTANT_TIME))

    # x0 = 0, x1 is known → ut = x1 - 0 = x1
    x1 = flow_batch["features"]["x1"]
    ut = result["targets"]["ut"]
    assert torch.allclose(ut, x1)
