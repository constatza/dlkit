"""Tests for build_model_from_checkpoint()'s state-dict reconciliation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dlkit.common.errors import WorkflowError
from dlkit.engine.inference.model_builder import build_model_from_checkpoint


def test_build_model_from_checkpoint_rejects_unexpected_key_for_non_fittable_model(
    simple_checkpoint: Path,
) -> None:
    """A genuinely bogus/extra top-level key must still fail strict loading.

    Regression guard for the Fittable buffer-preregistration path in
    build_model_from_checkpoint(): a plain torch.nn.Linear has no fit()/
    is_fitted() methods, so it must not be duck-typed as Fittable, and an
    unexpected key must still surface as a real error instead of being
    silently pre-registered and loaded.
    """
    checkpoint = torch.load(simple_checkpoint, weights_only=False)
    checkpoint["state_dict"] = {
        **checkpoint["state_dict"],
        "not_a_real_parameter": torch.zeros(3),
    }

    with pytest.raises(WorkflowError, match="not_a_real_parameter"):
        build_model_from_checkpoint(checkpoint)


def test_build_model_from_checkpoint_rejects_unexpected_nested_key_for_non_fittable_model(
    simple_checkpoint: Path,
) -> None:
    """A dotted (nested-submodule-shaped) unexpected key must still fail cleanly.

    nn.Module.register_buffer rejects any name containing "." outright, so a
    naive pre-registration loop would crash with a raw KeyError instead of
    the WorkflowError build_model_from_checkpoint() is documented to raise.
    """
    checkpoint = torch.load(simple_checkpoint, weights_only=False)
    checkpoint["state_dict"] = {
        **checkpoint["state_dict"],
        "encoder.norm.weight": torch.zeros(3),
    }

    with pytest.raises(WorkflowError, match="encoder.norm.weight"):
        build_model_from_checkpoint(checkpoint)
