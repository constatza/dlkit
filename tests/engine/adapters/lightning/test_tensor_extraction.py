"""Tests for as_flat_tensor — normalizes Tensor/TensorDict predict_step outputs.

predict_step legitimately wraps 'predictions'/'targets' in a nested TensorDict
keyed by target name, even for single-target regression models. as_flat_tensor
extracts the sole leaf tensor for the common single-target case.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from dlkit.engine.adapters.lightning.tensor_extraction import as_flat_tensor

BATCH_SIZE = 4


def test_bare_tensor_returned_as_is() -> None:
    """A plain torch.Tensor is returned unchanged."""
    tensor = torch.randn(BATCH_SIZE)
    assert as_flat_tensor(tensor) is tensor


def test_single_key_tensordict_extracts_leaf() -> None:
    """A single-key TensorDict yields its sole leaf tensor."""
    leaf = torch.randn(BATCH_SIZE)
    td = TensorDict({"y": leaf}, batch_size=[BATCH_SIZE])
    assert as_flat_tensor(td) is leaf


def test_multi_key_tensordict_returns_none() -> None:
    """A multi-key TensorDict has no unambiguous scalar and returns None."""
    td = TensorDict(
        {"y1": torch.randn(BATCH_SIZE), "y2": torch.randn(BATCH_SIZE)},
        batch_size=[BATCH_SIZE],
    )
    assert as_flat_tensor(td) is None


def test_non_tensor_leaf_returns_none() -> None:
    """A single-key Mapping whose leaf isn't a tensor returns None."""
    assert as_flat_tensor({"y": "not_a_tensor"}) is None


def test_unrelated_type_returns_none() -> None:
    """A value that's neither Tensor nor Mapping returns None."""
    assert as_flat_tensor(object()) is None
