"""Test suite for shape value objects.

This module tests the core value objects: ShapeEntry and ShapeData.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dlkit.core.shape_specs import ModelFamily, ShapeData, ShapeEntry, ShapeSource


class TestShapeEntry:
    """Test cases for ShapeEntry value object."""

    def test_shape_entry_creation_valid(self):
        """Test creation of valid shape entry."""
        entry = ShapeEntry(name="x", dimensions=(10, 20, 30))
        assert entry.name == "x"
        assert entry.dimensions == (10, 20, 30)

    def test_shape_entry_creation_single_dimension(self):
        """Test creation with single dimension."""
        entry = ShapeEntry(name="y", dimensions=(5,))
        assert entry.name == "y"
        assert entry.dimensions == (5,)

    def test_shape_entry_invalid_name_empty(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            ShapeEntry(name="", dimensions=(10,))

    def test_shape_entry_invalid_name_whitespace(self):
        """Test that whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            ShapeEntry(name="   ", dimensions=(10,))

    def test_shape_entry_invalid_dimensions_not_tuple(self):
        """Test that non-tuple dimensions raise ValueError."""
        with pytest.raises(ValueError, match="must be tuple"):
            ShapeEntry(name="x", dimensions=[10, 20])  # List instead of tuple

    def test_shape_entry_invalid_dimensions_empty(self):
        """Test that empty dimensions raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ShapeEntry(name="x", dimensions=())

    def test_shape_entry_invalid_dimensions_negative(self):
        """Test that negative dimensions raise ValueError."""
        with pytest.raises(ValueError, match="must be positive integer"):
            ShapeEntry(name="x", dimensions=(10, -5, 20))

    def test_shape_entry_invalid_dimensions_zero(self):
        """Test that zero dimensions raise ValueError."""
        with pytest.raises(ValueError, match="must be positive integer"):
            ShapeEntry(name="x", dimensions=(10, 0, 20))

    def test_shape_entry_invalid_dimensions_non_integer(self):
        """Test that non-integer dimensions raise ValueError."""
        with pytest.raises(ValueError, match="must be positive integer"):
            ShapeEntry(name="x", dimensions=(10.5, 20))

    def test_shape_entry_string_representation(self):
        """Test string representation."""
        entry = ShapeEntry(name="features", dimensions=(100, 50))
        assert str(entry) == "features=(100, 50)"

    @given(
        name=st.text(min_size=1).filter(lambda x: x.strip()),
        dimensions=st.tuples(st.integers(min_value=1, max_value=1000)).filter(lambda x: len(x) > 0),
    )
    def test_shape_entry_property_based(self, name, dimensions):
        """Property-based test for ShapeEntry creation."""
        entry = ShapeEntry(name=name.strip(), dimensions=dimensions)
        assert entry.name == name.strip()
        assert entry.dimensions == dimensions
        assert all(isinstance(d, int) and d > 0 for d in entry.dimensions)


class TestShapeData:
    """Test cases for ShapeData value object."""

    @pytest.fixture
    def sample_entries(self):
        """Sample entries for testing."""
        return {
            "x": ShapeEntry(name="x", dimensions=(10, 20)),
            "y": ShapeEntry(name="y", dimensions=(5,)),
            "features": ShapeEntry(name="features", dimensions=(100, 50, 25)),
        }

    def test_shape_data_creation_valid(self, sample_entries):
        """Test creation of valid shape data."""
        shape_data = ShapeData(
            entries=sample_entries,
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
            default_input="x",
            default_output="y",
        )

        assert shape_data.entries == sample_entries
        assert shape_data.model_family == ModelFamily.DLKIT_NN
        assert shape_data.source == ShapeSource.TRAINING_DATASET
        assert shape_data.default_input == "x"
        assert shape_data.default_output == "y"

    def test_shape_data_creation_minimal(self, sample_entries):
        """Test creation with minimal parameters."""
        shape_data = ShapeData(
            entries=sample_entries,
            model_family=ModelFamily.GRAPH,
            source=ShapeSource.CHECKPOINT_METADATA,
        )

        assert shape_data.entries == sample_entries
        assert shape_data.model_family == ModelFamily.GRAPH
        assert shape_data.source == ShapeSource.CHECKPOINT_METADATA
        assert shape_data.default_input is None
        assert shape_data.default_output is None

    def test_shape_data_invalid_entries_not_dict(self):
        """Test that non-dict entries raise ValueError."""
        with pytest.raises(ValueError, match="must be dictionary"):
            ShapeData(
                entries=["x", "y"],  # List instead of dict
                model_family=ModelFamily.DLKIT_NN,
                source=ShapeSource.TRAINING_DATASET,
            )

    def test_shape_data_invalid_model_family(self):
        """Test that invalid model family raises ValueError."""
        with pytest.raises(ValueError, match="must be ModelFamily enum"):
            ShapeData(
                entries={},
                model_family="invalid",  # String instead of enum
                source=ShapeSource.TRAINING_DATASET,
            )

    def test_shape_data_invalid_source(self):
        """Test that invalid source raises ValueError."""
        with pytest.raises(ValueError, match="must be ShapeSource enum"):
            ShapeData(
                entries={},
                model_family=ModelFamily.DLKIT_NN,
                source="invalid",  # String instead of enum
            )

    def test_shape_data_invalid_default_input_not_found(self, sample_entries):
        """Test that non-existent default input raises ValueError."""
        with pytest.raises(ValueError, match="Default input 'nonexistent' not found"):
            ShapeData(
                entries=sample_entries,
                model_family=ModelFamily.DLKIT_NN,
                source=ShapeSource.TRAINING_DATASET,
                default_input="nonexistent",
            )

    def test_shape_data_invalid_default_output_not_found(self, sample_entries):
        """Test that non-existent default output raises ValueError."""
        with pytest.raises(ValueError, match="Default output 'nonexistent' not found"):
            ShapeData(
                entries=sample_entries,
                model_family=ModelFamily.DLKIT_NN,
                source=ShapeSource.TRAINING_DATASET,
                default_output="nonexistent",
            )

    def test_shape_data_invalid_entry_type(self):
        """Test that non-ShapeEntry values raise ValueError."""
        with pytest.raises(ValueError, match="must be ShapeEntry"):
            ShapeData(
                entries={"x": (10, 20)},  # Tuple instead of ShapeEntry
                model_family=ModelFamily.DLKIT_NN,
                source=ShapeSource.TRAINING_DATASET,
            )

    def test_shape_data_invalid_entry_name_mismatch(self):
        """Test that mismatched entry names raise ValueError."""
        entry = ShapeEntry(name="y", dimensions=(10,))
        with pytest.raises(ValueError, match="Entry name mismatch"):
            ShapeData(
                entries={"x": entry},  # Key 'x' but entry name is 'y'
                model_family=ModelFamily.DLKIT_NN,
                source=ShapeSource.TRAINING_DATASET,
            )

    def test_shape_data_has_entry(self, sample_entries):
        """Test has_entry method."""
        shape_data = ShapeData(
            entries=sample_entries,
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )

        assert shape_data.has_entry("x")
        assert shape_data.has_entry("y")
        assert shape_data.has_entry("features")
        assert not shape_data.has_entry("nonexistent")

    def test_shape_data_get_entry(self, sample_entries):
        """Test get_entry method."""
        shape_data = ShapeData(
            entries=sample_entries,
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )

        entry = shape_data.get_entry("x")
        assert entry is not None
        assert entry.name == "x"
        assert entry.dimensions == (10, 20)

        assert shape_data.get_entry("nonexistent") is None

    def test_shape_data_get_dimensions(self, sample_entries):
        """Test get_dimensions method."""
        shape_data = ShapeData(
            entries=sample_entries,
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )

        assert shape_data.get_dimensions("x") == (10, 20)
        assert shape_data.get_dimensions("y") == (5,)
        assert shape_data.get_dimensions("features") == (100, 50, 25)
        assert shape_data.get_dimensions("nonexistent") is None

    def test_shape_data_entry_names(self, sample_entries):
        """Test entry_names method."""
        shape_data = ShapeData(
            entries=sample_entries,
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )

        names = shape_data.entry_names()
        assert names == {"x", "y", "features"}

    def test_shape_data_is_empty(self):
        """Test is_empty method."""
        empty_data = ShapeData(
            entries={}, model_family=ModelFamily.EXTERNAL, source=ShapeSource.DEFAULT_FALLBACK
        )
        assert empty_data.is_empty()

        non_empty_data = ShapeData(
            entries={"x": ShapeEntry(name="x", dimensions=(10,))},
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )
        assert not non_empty_data.is_empty()

    def test_shape_data_with_defaults(self, sample_entries):
        """Test with_defaults method."""
        original = ShapeData(
            entries=sample_entries,
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )

        updated = original.with_defaults(default_input="features", default_output="y")

        assert updated.entries == original.entries
        assert updated.model_family == original.model_family
        assert updated.source == original.source
        assert updated.default_input == "features"
        assert updated.default_output == "y"

        # Original should be unchanged
        assert original.default_input is None
        assert original.default_output is None

    def test_shape_data_len(self, sample_entries):
        """Test __len__ method."""
        shape_data = ShapeData(
            entries=sample_entries,
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )

        assert len(shape_data) == 3

    def test_shape_data_string_representation(self, sample_entries):
        """Test string representation."""
        shape_data = ShapeData(
            entries=sample_entries,
            model_family=ModelFamily.GRAPH,
            source=ShapeSource.CHECKPOINT_METADATA,
        )

        str_repr = str(shape_data)
        assert "ShapeData" in str_repr
        assert "graph" in str_repr
        assert "checkpoint_metadata" in str_repr

    @given(
        model_family=st.sampled_from(list(ModelFamily)), source=st.sampled_from(list(ShapeSource))
    )
    def test_shape_data_property_based(self, model_family, source):
        """Property-based test for ShapeData creation."""
        entries = {"test": ShapeEntry(name="test", dimensions=(10, 20))}

        shape_data = ShapeData(entries=entries, model_family=model_family, source=source)

        assert shape_data.model_family == model_family
        assert shape_data.source == source
        assert len(shape_data) == 1
        assert shape_data.has_entry("test")
