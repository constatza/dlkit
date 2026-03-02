"""Tests for ShapeSpec value object."""

import pytest
from dlkit.core.shape_specs import create_shape_spec, ModelFamily, ShapeSource


class TestShapeSpec:
    """Test suite for ShapeSpec value object."""

    def test_basic_construction(self):
        """Test basic ShapeSpec construction with validation."""
        shape_spec = create_shape_spec(
            shapes={"x": (10, 5), "y": (5,)},
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )

        assert shape_spec.get_shape("x") == (10, 5)
        assert shape_spec.get_shape("y") == (5,)
        assert shape_spec.has_shape("x")
        assert shape_spec.has_shape("y")

    def test_validation_invalid_data_type(self):
        """Test that invalid data types are rejected."""
        with pytest.raises(AttributeError):
            create_shape_spec(
                shapes="invalid",  # type: ignore
                model_family=ModelFamily.DLKIT_NN,
                source=ShapeSource.TRAINING_DATASET,
            )
