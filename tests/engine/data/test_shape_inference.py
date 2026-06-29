"""Tests for entry shape inference, in particular the batch-axis convention
shared between feature and target propagation."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from dlkit.engine.data.shape_inference import infer_entry_shapes
from dlkit.infrastructure.config.entry_types import ValueEntry
from dlkit.infrastructure.config.transform_settings import TransformSettings


def test_unsqueeze_on_a_feature_matches_real_batched_forward() -> None:
    """A feature with an ``Unsqueeze(dim=1)`` transform must resolve to the
    same per-sample shape that a real batched tensor would produce, mirroring
    how query-based models (e.g. DeepONet's trunk) consume it."""
    trunk = ValueEntry(name="trunk", transforms=[TransformSettings(name="Unsqueeze", dim=1)])
    sample = TensorDict(
        {"features": TensorDict({"trunk": torch.randn(4)}, batch_size=[])}, batch_size=[]
    )

    context = infer_entry_shapes([trunk], [], sample)

    batch_size = 3
    real_batched = torch.randn(batch_size, 4).unsqueeze(1)
    assert context.input_shapes["trunk"] == tuple(real_batched.shape[1:])


def test_feature_and_target_unsqueeze_resolve_identically() -> None:
    """Features and targets must share the same batch-axis convention so an
    identical transform on both sides resolves to the same shape."""
    branch = ValueEntry(name="branch", transforms=[TransformSettings(name="Unsqueeze", dim=1)])
    target = ValueEntry(name="y", transforms=[TransformSettings(name="Unsqueeze", dim=1)])
    sample = TensorDict(
        {
            "features": TensorDict({"branch": torch.randn(5)}, batch_size=[]),
            "targets": TensorDict({"y": torch.randn(5)}, batch_size=[]),
        },
        batch_size=[],
    )

    context = infer_entry_shapes([branch], [target], sample)

    assert context.input_shapes["branch"] == context.output_shapes["y"] == (1, 5)
