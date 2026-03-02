"""Unit tests for dlkit.tools.utils.tensordict_utils."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from dlkit.tools.utils import tensordict_to_numpy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_td() -> TensorDict:
    """TensorDict with flat Tensor leaves."""
    return TensorDict(
        {
            "predictions": torch.rand(5, 2),
            "targets": torch.rand(5, 1),
            "latents": torch.zeros(5, 0),
        },
        batch_size=5,
    )


@pytest.fixture
def nested_td() -> TensorDict:
    """TensorDict where 'targets' is itself a TensorDict with two named entries."""
    return TensorDict(
        {
            "predictions": torch.rand(5, 2),
            "targets": TensorDict({"y": torch.rand(5, 1), "z": torch.rand(5, 3)}, batch_size=5),
            "latents": torch.zeros(5, 0),
        },
        batch_size=5,
    )


# ---------------------------------------------------------------------------
# Tests — no-key (convert all)
# ---------------------------------------------------------------------------


class TestTensordictToNumpyNoKeys:
    """Behaviour when no key filter is supplied."""

    def test_all_flat_leaves_converted(self, flat_td):
        """Every leaf Tensor becomes an np.ndarray."""
        result = tensordict_to_numpy(flat_td)

        assert isinstance(result["predictions"], np.ndarray)
        assert isinstance(result["targets"], np.ndarray)
        assert isinstance(result["latents"], np.ndarray)

    def test_shapes_preserved(self, flat_td):
        """Array shapes match the original Tensor shapes."""
        result = tensordict_to_numpy(flat_td)

        assert result["predictions"].shape == (5, 2)
        assert result["targets"].shape == (5, 1)
        assert result["latents"].shape == (5, 0)

    def test_nested_structure_preserved(self, nested_td):
        """Nested TensorDict becomes a nested dict; leaves are np.ndarray."""
        result = tensordict_to_numpy(nested_td)

        assert isinstance(result["targets"], dict)
        assert isinstance(result["targets"]["y"], np.ndarray)
        assert isinstance(result["targets"]["z"], np.ndarray)
        assert result["targets"]["y"].shape == (5, 1)
        assert result["targets"]["z"].shape == (5, 3)


# ---------------------------------------------------------------------------
# Tests — flat key selection
# ---------------------------------------------------------------------------


class TestTensordictToNumpyFlatKey:
    """Behaviour when one or more flat string keys are supplied."""

    def test_single_flat_key(self, flat_td):
        """Only the requested key appears in the result."""
        result = tensordict_to_numpy(flat_td, "predictions")

        assert set(result.keys()) == {"predictions"}
        assert isinstance(result["predictions"], np.ndarray)

    def test_multiple_flat_keys(self, flat_td):
        """All requested keys are present; unrequested keys are absent."""
        result = tensordict_to_numpy(flat_td, "predictions", "targets")

        assert set(result.keys()) == {"predictions", "targets"}
        assert "latents" not in result


# ---------------------------------------------------------------------------
# Tests — nested key path selection
# ---------------------------------------------------------------------------


class TestTensordictToNumpyNestedKey:
    """Behaviour when tuple nested key paths are supplied."""

    def test_nested_path_selects_single_leaf(self, nested_td):
        """Tuple path selects one leaf, dropping sibling keys inside the sub-TensorDict."""
        result = tensordict_to_numpy(nested_td, ("targets", "y"))

        assert set(result.keys()) == {"targets"}
        assert set(result["targets"].keys()) == {"y"}
        assert isinstance(result["targets"]["y"], np.ndarray)
        assert result["targets"]["y"].shape == (5, 1)

    def test_nested_path_omits_sibling(self, nested_td):
        """The sibling leaf 'z' is absent when only 'y' is requested."""
        result = tensordict_to_numpy(nested_td, ("targets", "y"))
        assert "z" not in result["targets"]

    def test_mixed_flat_and_nested_keys(self, nested_td):
        """Flat key and nested path can be combined in a single call."""
        result = tensordict_to_numpy(nested_td, "predictions", ("targets", "y"))

        assert set(result.keys()) == {"predictions", "targets"}
        assert isinstance(result["predictions"], np.ndarray)
        assert isinstance(result["targets"]["y"], np.ndarray)
        assert "z" not in result["targets"]
