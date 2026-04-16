"""Test suite for shape specifications.

This module tests the specification pattern implementation for validation.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dlkit.domain.shapes import (
    DimensionRangeSpecification,
    ModelFamily,
    ModelFamilyCompatibilitySpecification,
    PositiveDimensionsSpecification,
    RequiredEntriesSpecification,
    ShapeData,
    ShapeEntry,
    ShapeSource,
    ShapeSpecificationBuilder,
    ShapeValidationEngine,
    ValidationResult,
)


class TestValidationResult:
    """Test cases for ValidationResult."""

    def test_validation_result_success(self):
        """Test successful validation result."""
        result = ValidationResult.success()
        assert result.is_valid
        assert result.errors == ()
        assert result.warnings == ()

    def test_validation_result_failure(self):
        """Test failed validation result."""
        errors = ["Error 1", "Error 2"]
        result = ValidationResult.failure(errors)
        assert not result.is_valid
        assert result.errors == tuple(errors)
        assert result.warnings == ()

    def test_validation_result_add_error(self):
        """Test adding error to validation result."""
        result = ValidationResult.success()
        result = result.add_error("Test error")
        assert not result.is_valid
        assert "Test error" in result.errors

    def test_validation_result_add_warning(self):
        """Test adding warning to validation result."""
        result = ValidationResult.success()
        result = result.add_warning("Test warning")
        assert result.is_valid  # Still valid with warnings
        assert "Test warning" in result.warnings


class TestRequiredEntriesSpecification:
    """Test cases for RequiredEntriesSpecification."""

    @pytest.fixture
    def sample_shape_data(self):
        """Sample shape data for testing."""
        entries = {
            "x": ShapeEntry(name="x", dimensions=(10, 20)),
            "y": ShapeEntry(name="y", dimensions=(5,)),
            "features": ShapeEntry(name="features", dimensions=(100,)),
        }
        return ShapeData(
            entries=entries, model_family=ModelFamily.DLKIT_NN, source=ShapeSource.TRAINING_DATASET
        )

    def test_required_entries_all_present(self, sample_shape_data):
        """Test when all required entries are present."""
        spec = RequiredEntriesSpecification({"x", "y"})
        result = spec.is_satisfied_by(sample_shape_data)
        assert result.is_valid
        assert result.errors == ()

    def test_required_entries_some_missing(self, sample_shape_data):
        """Test when some required entries are missing."""
        spec = RequiredEntriesSpecification({"x", "y", "z", "w"})
        result = spec.is_satisfied_by(sample_shape_data)
        assert not result.is_valid
        assert len(result.errors) == 2
        assert any("'z'" in error for error in result.errors)
        assert any("'w'" in error for error in result.errors)

    def test_required_entries_empty_set(self, sample_shape_data):
        """Test with empty required entries set."""
        spec = RequiredEntriesSpecification(set())
        result = spec.is_satisfied_by(sample_shape_data)
        assert result.is_valid


class TestPositiveDimensionsSpecification:
    """Test cases for PositiveDimensionsSpecification."""

    def test_positive_dimensions_valid(self):
        """Test with all positive dimensions."""
        entries = {
            "x": ShapeEntry(name="x", dimensions=(10, 20, 30)),
            "y": ShapeEntry(name="y", dimensions=(5,)),
        }
        shape_data = ShapeData(
            entries=entries, model_family=ModelFamily.DLKIT_NN, source=ShapeSource.TRAINING_DATASET
        )

        spec = PositiveDimensionsSpecification()
        result = spec.is_satisfied_by(shape_data)
        assert result.is_valid

    def test_positive_dimensions_empty_data(self):
        """Test with empty shape data."""
        shape_data = ShapeData(
            entries={}, model_family=ModelFamily.EXTERNAL, source=ShapeSource.DEFAULT_FALLBACK
        )

        spec = PositiveDimensionsSpecification()
        result = spec.is_satisfied_by(shape_data)
        assert result.is_valid  # Empty data is valid


class TestModelFamilyCompatibilitySpecification:
    """Test cases for ModelFamilyCompatibilitySpecification."""

    def test_dlkit_nn_compatibility_with_xy(self):
        """Test DLKIT_NN compatibility with x/y entries."""
        entries = {
            "x": ShapeEntry(name="x", dimensions=(10,)),
            "y": ShapeEntry(name="y", dimensions=(5,)),
        }
        shape_data = ShapeData(
            entries=entries, model_family=ModelFamily.DLKIT_NN, source=ShapeSource.TRAINING_DATASET
        )

        spec = ModelFamilyCompatibilitySpecification(ModelFamily.DLKIT_NN)
        result = spec.is_satisfied_by(shape_data)
        assert result.is_valid
        assert len(result.warnings) == 0

    def test_dlkit_nn_compatibility_missing_x(self):
        """Test DLKIT_NN compatibility with missing x entry."""
        entries = {"y": ShapeEntry(name="y", dimensions=(5,))}
        shape_data = ShapeData(
            entries=entries, model_family=ModelFamily.DLKIT_NN, source=ShapeSource.TRAINING_DATASET
        )

        spec = ModelFamilyCompatibilitySpecification(ModelFamily.DLKIT_NN)
        result = spec.is_satisfied_by(shape_data)
        assert result.is_valid  # Warnings don't make it invalid
        assert any("'x' entry" in warning for warning in result.warnings)

    def test_graph_compatibility_with_x(self):
        """Test GRAPH compatibility with x entry."""
        entries = {
            "x": ShapeEntry(name="x", dimensions=(100, 50)),
            "edge_index": ShapeEntry(name="edge_index", dimensions=(2, 1000)),
        }
        shape_data = ShapeData(
            entries=entries, model_family=ModelFamily.GRAPH, source=ShapeSource.GRAPH_DATASET
        )

        spec = ModelFamilyCompatibilitySpecification(ModelFamily.GRAPH)
        result = spec.is_satisfied_by(shape_data)
        assert result.is_valid
        assert len(result.warnings) == 0

    def test_external_compatibility_with_shapes(self):
        """Test EXTERNAL compatibility with shape data."""
        entries = {"x": ShapeEntry(name="x", dimensions=(10,))}
        shape_data = ShapeData(
            entries=entries, model_family=ModelFamily.EXTERNAL, source=ShapeSource.CONFIGURATION
        )

        spec = ModelFamilyCompatibilitySpecification(ModelFamily.EXTERNAL)
        result = spec.is_satisfied_by(shape_data)
        assert result.is_valid  # Valid but with warning
        assert any("don't use shape" in warning for warning in result.warnings)


class TestDimensionRangeSpecification:
    """Test cases for DimensionRangeSpecification."""

    def test_dimension_range_within_bounds(self):
        """Test with dimensions within range."""
        entries = {
            "x": ShapeEntry(name="x", dimensions=(10, 20)),  # 2 dimensions
            "y": ShapeEntry(name="y", dimensions=(5,)),  # 1 dimension
        }
        shape_data = ShapeData(
            entries=entries, model_family=ModelFamily.DLKIT_NN, source=ShapeSource.TRAINING_DATASET
        )

        spec = DimensionRangeSpecification(min_dimensions=1, max_dimensions=3)
        result = spec.is_satisfied_by(shape_data)
        assert result.is_valid

    def test_dimension_range_below_minimum(self):
        """Test with dimensions below minimum."""
        # Note: Can't actually create ShapeEntry with 0 dimensions due to validation
        # So we test the specification logic with mock data
        entries = {
            "x": ShapeEntry(name="x", dimensions=(10,))  # 1 dimension
        }
        shape_data = ShapeData(
            entries=entries, model_family=ModelFamily.DLKIT_NN, source=ShapeSource.TRAINING_DATASET
        )

        spec = DimensionRangeSpecification(min_dimensions=2, max_dimensions=5)
        result = spec.is_satisfied_by(shape_data)
        assert not result.is_valid
        assert any("minimum required: 2" in error for error in result.errors)

    def test_dimension_range_above_maximum(self):
        """Test with dimensions above maximum."""
        entries = {
            "x": ShapeEntry(name="x", dimensions=(10, 20, 30, 40, 50, 60))  # 6 dimensions
        }
        shape_data = ShapeData(
            entries=entries, model_family=ModelFamily.DLKIT_NN, source=ShapeSource.TRAINING_DATASET
        )

        spec = DimensionRangeSpecification(min_dimensions=1, max_dimensions=3)
        result = spec.is_satisfied_by(shape_data)
        assert not result.is_valid
        assert any("maximum allowed: 3" in error for error in result.errors)


class TestCompositeSpecifications:
    """Test cases for composite specifications (AND, OR, NOT)."""

    @pytest.fixture
    def valid_shape_data(self):
        """Valid shape data for testing."""
        entries = {
            "x": ShapeEntry(name="x", dimensions=(10, 20)),
            "y": ShapeEntry(name="y", dimensions=(5,)),
        }
        return ShapeData(
            entries=entries, model_family=ModelFamily.DLKIT_NN, source=ShapeSource.TRAINING_DATASET
        )

    def test_and_specification_both_valid(self, valid_shape_data):
        """Test AND specification when both specs are valid."""
        spec1 = RequiredEntriesSpecification({"x"})
        spec2 = DimensionRangeSpecification(min_dimensions=1, max_dimensions=3)

        combined_spec = spec1.and_(spec2)
        result = combined_spec.is_satisfied_by(valid_shape_data)
        assert result.is_valid

    def test_and_specification_one_invalid(self, valid_shape_data):
        """Test AND specification when one spec is invalid."""
        spec1 = RequiredEntriesSpecification({"x", "z"})  # z is missing
        spec2 = DimensionRangeSpecification(min_dimensions=1, max_dimensions=3)

        combined_spec = spec1.and_(spec2)
        result = combined_spec.is_satisfied_by(valid_shape_data)
        assert not result.is_valid
        assert any("'z'" in error for error in result.errors)

    def test_or_specification_one_valid(self, valid_shape_data):
        """Test OR specification when one spec is valid."""
        spec1 = RequiredEntriesSpecification({"z"})  # Invalid - z is missing
        spec2 = RequiredEntriesSpecification({"x"})  # Valid - x is present

        combined_spec = spec1.or_(spec2)
        result = combined_spec.is_satisfied_by(valid_shape_data)
        assert result.is_valid

    def test_or_specification_both_invalid(self, valid_shape_data):
        """Test OR specification when both specs are invalid."""
        spec1 = RequiredEntriesSpecification({"z"})  # Invalid - z is missing
        spec2 = RequiredEntriesSpecification({"w"})  # Invalid - w is missing

        combined_spec = spec1.or_(spec2)
        result = combined_spec.is_satisfied_by(valid_shape_data)
        assert not result.is_valid

    def test_not_specification_valid_input(self, valid_shape_data):
        """Test NOT specification with valid input."""
        spec = RequiredEntriesSpecification({"x"})  # Valid spec
        not_spec = spec.not_()

        result = not_spec.is_satisfied_by(valid_shape_data)
        assert not result.is_valid  # NOT of valid spec should be invalid

    def test_not_specification_invalid_input(self, valid_shape_data):
        """Test NOT specification with invalid input."""
        spec = RequiredEntriesSpecification({"z"})  # Invalid spec - z is missing
        not_spec = spec.not_()

        result = not_spec.is_satisfied_by(valid_shape_data)
        assert result.is_valid  # NOT of invalid spec should be valid


class TestShapeValidationEngine:
    """Test cases for ShapeValidationEngine."""

    @pytest.fixture
    def validation_engine(self):
        """Validation engine for testing."""
        return ShapeValidationEngine()

    def test_validation_engine_dlkit_nn(self, validation_engine):
        """Test validation engine with DLKIT_NN model family."""
        entries = {
            "x": ShapeEntry(name="x", dimensions=(10, 20)),
            "y": ShapeEntry(name="y", dimensions=(5,)),
        }
        shape_data = ShapeData(
            entries=entries, model_family=ModelFamily.DLKIT_NN, source=ShapeSource.TRAINING_DATASET
        )

        result = validation_engine.validate(shape_data)
        assert result.is_valid

    def test_validation_engine_graph(self, validation_engine):
        """Test validation engine with GRAPH model family."""
        entries = {"x": ShapeEntry(name="x", dimensions=(100, 50))}
        shape_data = ShapeData(
            entries=entries, model_family=ModelFamily.GRAPH, source=ShapeSource.GRAPH_DATASET
        )

        result = validation_engine.validate(shape_data)
        assert result.is_valid

    def test_validation_engine_custom_specification(self, validation_engine):
        """Test validation engine with custom specification."""
        entries = {"x": ShapeEntry(name="x", dimensions=(10,))}
        shape_data = ShapeData(
            entries=entries, model_family=ModelFamily.DLKIT_NN, source=ShapeSource.TRAINING_DATASET
        )

        # Custom spec requiring both x and y
        custom_spec = RequiredEntriesSpecification({"x", "y"})
        result = validation_engine.validate_with_spec(shape_data, custom_spec)
        assert not result.is_valid
        assert any("'y'" in error for error in result.errors)


class TestShapeSpecificationBuilder:
    """Test cases for ShapeSpecificationBuilder."""

    def test_builder_single_specification(self):
        """Test builder with single specification."""
        builder = ShapeSpecificationBuilder()
        spec = builder.require_entries({"x", "y"}).build()

        entries = {
            "x": ShapeEntry(name="x", dimensions=(10,)),
            "y": ShapeEntry(name="y", dimensions=(5,)),
        }
        shape_data = ShapeData(
            entries=entries, model_family=ModelFamily.DLKIT_NN, source=ShapeSource.TRAINING_DATASET
        )

        result = spec.is_satisfied_by(shape_data)
        assert result.is_valid

    def test_builder_multiple_specifications(self):
        """Test builder with multiple specifications."""
        builder = ShapeSpecificationBuilder()
        spec = builder.require_entries({"x"}).positive_dimensions().dimension_range(1, 3).build()

        entries = {"x": ShapeEntry(name="x", dimensions=(10, 20))}
        shape_data = ShapeData(
            entries=entries, model_family=ModelFamily.DLKIT_NN, source=ShapeSource.TRAINING_DATASET
        )

        result = spec.is_satisfied_by(shape_data)
        assert result.is_valid

    def test_builder_empty_raises_error(self):
        """Test builder with no specifications raises error."""
        builder = ShapeSpecificationBuilder()
        with pytest.raises(ValueError, match="no rules"):
            builder.build()

    @given(
        required_entries=st.sets(
            st.text(min_size=1).filter(
                lambda x: x.strip() and x.isascii() and x.replace(" ", "_") == x
            ),
            min_size=1,
            max_size=3,
        ),
        min_dims=st.integers(min_value=1, max_value=3),
        max_dims=st.integers(min_value=4, max_value=10),
    )
    def test_builder_property_based(self, required_entries, min_dims, max_dims):
        """Property-based test for specification builder."""
        builder = ShapeSpecificationBuilder()
        spec = builder.require_entries(required_entries).dimension_range(min_dims, max_dims).build()

        # Create valid shape data
        entries = {}
        for entry_name in required_entries:
            # Create dimensions within range
            num_dims = min_dims + 1 if min_dims < max_dims else min_dims
            dimensions = tuple(range(1, num_dims + 1))
            entries[entry_name] = ShapeEntry(name=entry_name, dimensions=dimensions)

        shape_data = ShapeData(
            entries=entries, model_family=ModelFamily.DLKIT_NN, source=ShapeSource.TRAINING_DATASET
        )

        result = spec.is_satisfied_by(shape_data)
        assert result.is_valid
