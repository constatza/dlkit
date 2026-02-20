import numpy as np
import pytest
from dlkit.interfaces.api.domain.models import TrainingResult


@pytest.fixture
def triplet_batches() -> list:
    """Two prediction batches in triplet tuple format with multi-variable outputs."""
    f1_b0 = np.random.rand(4, 2)
    f2_b0 = np.random.rand(4, 3)
    t1_b0 = np.random.rand(4, 1)
    l1_b0 = np.random.rand(4, 5)

    f1_b1 = np.random.rand(2, 2)
    f2_b1 = np.random.rand(2, 3)
    t1_b1 = np.random.rand(2, 1)
    l1_b1 = np.random.rand(2, 5)

    return [
        ((f1_b0, f2_b0), (t1_b0,), (l1_b0,)),
        ((f1_b1, f2_b1), (t1_b1,), (l1_b1,)),
    ]


@pytest.fixture
def inhomogeneous_batches() -> list:
    """Two batches where one variable has irregular shapes."""
    f1_b0 = np.random.rand(2, 2)
    f2_b0 = [np.random.rand(2), np.random.rand(3)]  # irregular

    f1_b1 = np.random.rand(2, 2)
    f2_b1 = [np.random.rand(4)]

    return [
        ((f1_b0, f2_b0), (), ()),
        ((f1_b1, f2_b1), (), ()),
    ]


@pytest.fixture
def dict_batches() -> list:
    """Two prediction batches in dict format."""
    return [
        {"predictions": (np.array([1, 2]),), "targets": (np.array([0]),)},
        {"predictions": (np.array([3, 4]),), "targets": (np.array([1]),)},
    ]


@pytest.fixture
def training_result_factory():
    """Factory fixture that builds a TrainingResult with given predictions."""

    def _make(predictions):
        return TrainingResult(
            model_state=None,
            metrics={},
            artifacts={},
            duration_seconds=1.0,
            predictions=predictions,
        )

    return _make


class TestTrainingResultStacking:
    """Unit tests for TrainingResult.stacked and StackedResults."""

    def test_triplet_predictions_stacked_correctly(
        self, triplet_batches, training_result_factory
    ):
        """Multi-variable predictions are concatenated per position."""
        result = training_result_factory(triplet_batches)
        (f1_b0, f2_b0), *_ = triplet_batches[0]
        (f1_b1, f2_b1), *_ = triplet_batches[1]

        stacked_p = result.stacked.predictions
        assert isinstance(stacked_p, tuple)
        assert len(stacked_p) == 2
        assert np.array_equal(stacked_p[0], np.concatenate([f1_b0, f1_b1], axis=0))
        assert np.array_equal(stacked_p[1], np.concatenate([f2_b0, f2_b1], axis=0))

    def test_triplet_targets_stacked_correctly(
        self, triplet_batches, training_result_factory
    ):
        """Single-variable targets are returned as a plain array."""
        result = training_result_factory(triplet_batches)
        _, (t1_b0,), _ = triplet_batches[0]
        _, (t1_b1,), _ = triplet_batches[1]

        stacked_t = result.stacked.targets
        assert isinstance(stacked_t, np.ndarray)
        assert np.array_equal(stacked_t, np.concatenate([t1_b0, t1_b1], axis=0))

    def test_triplet_latents_stacked_correctly(
        self, triplet_batches, training_result_factory
    ):
        """Single-variable latents are returned as a plain array."""
        result = training_result_factory(triplet_batches)
        _, _, (l1_b0,) = triplet_batches[0]
        _, _, (l1_b1,) = triplet_batches[1]

        stacked_l = result.stacked.latents
        assert isinstance(stacked_l, np.ndarray)
        assert np.array_equal(stacked_l, np.concatenate([l1_b0, l1_b1], axis=0))

    def test_inhomogeneous_variable_falls_back_to_list(
        self, inhomogeneous_batches, training_result_factory
    ):
        """Irregular variable falls back to list; regular variable is concatenated."""
        result = training_result_factory(inhomogeneous_batches)
        (f1_b0, f2_b0), *_ = inhomogeneous_batches[0]
        (f1_b1, f2_b1), *_ = inhomogeneous_batches[1]

        stacked_p = result.stacked.predictions
        assert isinstance(stacked_p, tuple)
        assert len(stacked_p) == 2
        assert stacked_p[0].shape == (4, 2)
        assert isinstance(stacked_p[1], list)
        assert stacked_p[1][0] is f2_b0
        assert stacked_p[1][1] is f2_b1

    def test_dictionary_format(self, dict_batches, training_result_factory):
        """Dict-format batches are parsed correctly."""
        result = training_result_factory(dict_batches)

        assert np.array_equal(result.stacked.predictions, np.array([1, 2, 3, 4]))
        assert np.array_equal(result.stacked.targets, np.array([0, 1]))

    def test_empty_predictions_returns_empty_stacked(self, training_result_factory):
        """No prediction batches yields a StackedResults with all-None fields."""
        result = training_result_factory(None)

        assert result.stacked.predictions is None
        assert result.stacked.targets is None
        assert result.stacked.latents is None

    def test_stacked_is_cached(self, dict_batches, training_result_factory):
        """Accessing .stacked twice returns the identical object (cached_property)."""
        result = training_result_factory(dict_batches)
        assert result.stacked is result.stacked
