from typing import cast

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from dlkit.shared import TrainingResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def training_result_factory():
    """Factory fixture that builds a TrainingResult with given predictions."""

    def _make(predictions: list | None) -> TrainingResult:
        return TrainingResult(
            model_state=None,
            metrics={},
            artifacts={},
            duration_seconds=1.0,
            predictions=predictions,
        )

    return _make


@pytest.fixture
def tensordict_batches() -> list[TensorDict]:
    """Two batches: plain Tensor predictions/targets, zero-size latents sentinel."""
    preds_b0 = torch.rand(4, 2)
    targets_b0 = torch.rand(4, 1)
    preds_b1 = torch.rand(3, 2)
    targets_b1 = torch.rand(3, 1)
    return [
        TensorDict(
            {"predictions": preds_b0, "targets": targets_b0, "latents": torch.zeros(4, 0)},
            batch_size=4,
        ),
        TensorDict(
            {"predictions": preds_b1, "targets": targets_b1, "latents": torch.zeros(3, 0)},
            batch_size=3,
        ),
    ]


@pytest.fixture
def tensordict_with_latents_batches() -> list[TensorDict]:
    """Two batches that carry real latent representations."""
    preds_b0 = torch.rand(4, 2)
    targets_b0 = torch.rand(4, 1)
    latents_b0 = torch.rand(4, 5)
    preds_b1 = torch.rand(3, 2)
    targets_b1 = torch.rand(3, 1)
    latents_b1 = torch.rand(3, 5)
    return [
        TensorDict(
            {"predictions": preds_b0, "targets": targets_b0, "latents": latents_b0},
            batch_size=4,
        ),
        TensorDict(
            {"predictions": preds_b1, "targets": targets_b1, "latents": latents_b1},
            batch_size=3,
        ),
    ]


@pytest.fixture
def tensordict_nested_targets_batches() -> list[TensorDict]:
    """Batches where 'targets' is a nested TensorDict — real FlexibleDataset output."""
    preds_b0 = torch.rand(4, 2)
    preds_b1 = torch.rand(3, 2)
    return [
        TensorDict(
            {
                "predictions": preds_b0,
                "targets": TensorDict({"y": torch.rand(4, 1)}, batch_size=4),
                "latents": torch.zeros(4, 0),
            },
            batch_size=4,
        ),
        TensorDict(
            {
                "predictions": preds_b1,
                "targets": TensorDict({"y": torch.rand(3, 1)}, batch_size=3),
                "latents": torch.zeros(3, 0),
            },
            batch_size=3,
        ),
    ]


@pytest.fixture
def tensordict_nested_multi_targets_batches() -> list[TensorDict]:
    """Batches where 'targets' is a nested TensorDict with multiple named entries."""
    preds_b0 = torch.rand(4, 2)
    preds_b1 = torch.rand(3, 2)
    return [
        TensorDict(
            {
                "predictions": preds_b0,
                "targets": TensorDict({"y": torch.rand(4, 1), "z": torch.rand(4, 3)}, batch_size=4),
                "latents": torch.zeros(4, 0),
            },
            batch_size=4,
        ),
        TensorDict(
            {
                "predictions": preds_b1,
                "targets": TensorDict({"y": torch.rand(3, 1), "z": torch.rand(3, 3)}, batch_size=3),
                "latents": torch.zeros(3, 0),
            },
            batch_size=3,
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainingResultStacking:
    """Unit tests for TrainingResult.stacked."""

    def test_flat_predictions_and_targets_stacked(
        self, tensordict_batches, training_result_factory
    ):
        """Plain Tensor predictions and targets are concatenated across batches."""
        b0, b1 = tensordict_batches
        result = training_result_factory(tensordict_batches)

        stacked = result.stacked
        assert stacked is not None
        assert stacked["predictions"].shape == (7, 2)
        assert torch.allclose(stacked["predictions"][:4], b0["predictions"])
        assert torch.allclose(stacked["predictions"][4:], b1["predictions"])
        assert stacked["targets"].shape == (7, 1)

    def test_absent_latents_produce_zero_size_sentinel(
        self, tensordict_batches, training_result_factory
    ):
        """Batches without real latents have a (N, 0) sentinel after stacking."""
        result = training_result_factory(tensordict_batches)
        assert result.stacked["latents"].shape == (7, 0)

    def test_real_latents_stacked(self, tensordict_with_latents_batches, training_result_factory):
        """Real latent tensors are concatenated preserving their feature dimension."""
        b0, _ = tensordict_with_latents_batches
        result = training_result_factory(tensordict_with_latents_batches)

        stacked = result.stacked
        assert stacked["predictions"].shape == (7, 2)
        assert stacked["targets"].shape == (7, 1)
        assert stacked["latents"].shape == (7, 5)
        assert torch.allclose(stacked["latents"][:4], b0["latents"])

    def test_nested_single_target_stacked(
        self, tensordict_nested_targets_batches, training_result_factory
    ):
        """Nested TensorDict targets are preserved and stacked by key name."""
        b0, _ = tensordict_nested_targets_batches
        result = training_result_factory(tensordict_nested_targets_batches)

        stacked = result.stacked
        nested_targets = cast(TensorDict, stacked["targets"])
        assert stacked["predictions"].shape == (7, 2)
        assert nested_targets["y"].shape == (7, 1)
        assert torch.allclose(
            cast("torch.Tensor", nested_targets["y"][:4]),
            cast("torch.Tensor", cast(TensorDict, b0["targets"])["y"]),
        )
        assert stacked["latents"].shape == (7, 0)

    def test_nested_multi_target_stacked(
        self, tensordict_nested_multi_targets_batches, training_result_factory
    ):
        """All named entries in a nested targets TensorDict are individually stacked."""
        result = training_result_factory(tensordict_nested_multi_targets_batches)

        targets = cast(TensorDict, result.stacked["targets"])
        assert targets["y"].shape == (7, 1)
        assert targets["z"].shape == (7, 3)

    def test_empty_predictions_returns_none(self, training_result_factory):
        """No prediction batches yields None."""
        assert training_result_factory(None).stacked is None

    def test_stacked_is_cached(self, tensordict_batches, training_result_factory):
        """Accessing .stacked twice returns the identical TensorDict object."""
        result = training_result_factory(tensordict_batches)
        assert result.stacked is result.stacked


class TestToNumpy:
    """Unit tests for TrainingResult.to_numpy convenience method."""

    def test_no_predictions_returns_none(self, training_result_factory):
        """to_numpy returns None when there are no predictions."""
        assert training_result_factory(None).to_numpy() is None

    def test_converts_all_fields(self, tensordict_batches, training_result_factory):
        """Without keys, every field is converted."""
        result = training_result_factory(tensordict_batches)
        arrays = result.to_numpy()

        assert isinstance(arrays["predictions"], np.ndarray)
        assert isinstance(arrays["targets"], np.ndarray)
        assert arrays["predictions"].shape == (7, 2)

    def test_flat_key_selection(self, tensordict_batches, training_result_factory):
        """Single flat key returns only that field."""
        result = training_result_factory(tensordict_batches)
        arrays = result.to_numpy("predictions")

        assert set(arrays.keys()) == {"predictions"}
        assert isinstance(arrays["predictions"], np.ndarray)

    def test_nested_key_path(self, tensordict_nested_targets_batches, training_result_factory):
        """Tuple nested path selects a leaf, keeping only that sub-tree."""
        result = training_result_factory(tensordict_nested_targets_batches)
        arrays = result.to_numpy(("targets", "y"))

        assert set(arrays.keys()) == {"targets"}
        assert isinstance(arrays["targets"]["y"], np.ndarray)
        assert arrays["targets"]["y"].shape == (7, 1)

    def test_mixed_flat_and_nested_keys(
        self, tensordict_nested_targets_batches, training_result_factory
    ):
        """Flat key and nested path can be requested together."""
        result = training_result_factory(tensordict_nested_targets_batches)
        arrays = result.to_numpy("predictions", ("targets", "y"))

        assert set(arrays.keys()) == {"predictions", "targets"}
        assert isinstance(arrays["predictions"], np.ndarray)
        assert isinstance(arrays["targets"]["y"], np.ndarray)
