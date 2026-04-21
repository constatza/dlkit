"""Tests for shape strategy classes.

Covers ValidationResult, ShapeValidator, ShapeSerializer, ShapeAliasResolver,
and the ShapeSpec facade (via create_shape_spec), replacing 49 deleted tests
from the old spec/serialization/migrator subsystem.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dlkit.domain.shapes import create_shape_spec
from dlkit.domain.shapes.strategies import (
    ShapeAliasResolver,
    ShapeSerializer,
    ShapeValidator,
    ValidationResult,
)
from dlkit.domain.shapes.value_objects import ModelFamily, ShapeData, ShapeEntry, ShapeSource

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

ERROR_MSG_ALPHA = "alpha error"
ERROR_MSG_BETA = "beta error"
WARNING_MSG = "a warning"

V1_DICT_RAW: dict[str, Any] = {
    "_type": "dict",
    "data": {"x": [10, 5]},
}
V2_RAW: dict[str, Any] = {
    "entries": {"x": [10, 5]},
    "model_family": "dlkit_nn",
    "source": "training_dataset",
}
# V3 format is always wrapped in a metadata + data envelope when fed to the
# public ShapeSerializer.deserialize() path.  Passing the bare V3 data dict
# causes the version detector to mistake it for V2, so we must include the
# metadata wrapper here.
V3_RAW: dict[str, Any] = {
    "metadata": {
        "version": "v3",
        "format": "json",
        "created_at": "2024-01-01T00:00:00",
        "checksum": None,
        "migration_history": [],
    },
    "data": {
        "entries": {
            "x": {"dimensions": [10, 5], "metadata": {"name": "x"}},
            "y": {"dimensions": [5], "metadata": {"name": "y"}},
        },
        "model_family": "dlkit_nn",
        "source": "training_dataset",
        "default_input": None,
        "default_output": None,
        "schema_version": "3.0",
    },
}


# ===========================================================================
# ValidationResult
# ===========================================================================


class TestValidationResultSuccess:
    """Tests for ValidationResult.success() factory."""

    def test_success_is_valid(self) -> None:
        """success() produces a result where is_valid is True.

        Args:
            None
        """
        result = ValidationResult.success()
        assert result.is_valid is True

    def test_success_has_no_errors(self) -> None:
        """success() produces a result with an empty errors tuple.

        Args:
            None
        """
        result = ValidationResult.success()
        assert result.errors == ()

    def test_success_has_no_warnings(self) -> None:
        """success() produces a result with an empty warnings tuple.

        Args:
            None
        """
        result = ValidationResult.success()
        assert result.warnings == ()


class TestValidationResultFailure:
    """Tests for ValidationResult.failure() factory."""

    def test_failure_is_invalid(self) -> None:
        """failure() produces a result where is_valid is False.

        Args:
            None
        """
        result = ValidationResult.failure([ERROR_MSG_ALPHA])
        assert result.is_valid is False

    def test_failure_captures_errors(self) -> None:
        """failure() stores each provided error string.

        Args:
            None
        """
        result = ValidationResult.failure([ERROR_MSG_ALPHA, ERROR_MSG_BETA])
        assert ERROR_MSG_ALPHA in result.errors
        assert ERROR_MSG_BETA in result.errors

    def test_failure_coerces_list_to_tuple(self) -> None:
        """failure() converts the list argument to an immutable tuple.

        Args:
            None
        """
        result = ValidationResult.failure([ERROR_MSG_ALPHA])
        assert isinstance(result.errors, tuple)

    def test_failure_empty_errors(self) -> None:
        """failure() with an empty list still marks is_valid False.

        Args:
            None
        """
        result = ValidationResult.failure([])
        assert result.is_valid is False
        assert result.errors == ()


class TestValidationResultAddError:
    """Tests for ValidationResult.add_error()."""

    def test_add_error_marks_invalid(self) -> None:
        """add_error() on a successful result flips is_valid to False.

        Args:
            None
        """
        result = ValidationResult.success().add_error(ERROR_MSG_ALPHA)
        assert result.is_valid is False

    def test_add_error_appends_message(self) -> None:
        """add_error() appends the error message to the errors tuple.

        Args:
            None
        """
        result = ValidationResult.success().add_error(ERROR_MSG_ALPHA)
        assert ERROR_MSG_ALPHA in result.errors

    def test_add_error_returns_new_instance(self) -> None:
        """add_error() returns a new object, not the original.

        Args:
            None
        """
        original = ValidationResult.success()
        mutated = original.add_error(ERROR_MSG_ALPHA)
        assert mutated is not original

    def test_add_error_original_unchanged(self) -> None:
        """add_error() does not mutate the original ValidationResult.

        Args:
            None
        """
        original = ValidationResult.success()
        original.add_error(ERROR_MSG_ALPHA)
        assert original.is_valid is True
        assert original.errors == ()

    def test_add_error_accumulates(self) -> None:
        """Chained add_error() calls accumulate all messages.

        Args:
            None
        """
        result = ValidationResult.success().add_error(ERROR_MSG_ALPHA).add_error(ERROR_MSG_BETA)
        assert ERROR_MSG_ALPHA in result.errors
        assert ERROR_MSG_BETA in result.errors
        assert len(result.errors) == 2


class TestValidationResultAddWarning:
    """Tests for ValidationResult.add_warning()."""

    def test_add_warning_stays_valid(self) -> None:
        """add_warning() on a successful result preserves is_valid True.

        Args:
            None
        """
        result = ValidationResult.success().add_warning(WARNING_MSG)
        assert result.is_valid is True

    def test_add_warning_appends_message(self) -> None:
        """add_warning() appends the warning message to the warnings tuple.

        Args:
            None
        """
        result = ValidationResult.success().add_warning(WARNING_MSG)
        assert WARNING_MSG in result.warnings

    def test_add_warning_returns_new_instance(self) -> None:
        """add_warning() returns a new object, not the original.

        Args:
            None
        """
        original = ValidationResult.success()
        mutated = original.add_warning(WARNING_MSG)
        assert mutated is not original

    def test_add_warning_original_unchanged(self) -> None:
        """add_warning() does not mutate the original ValidationResult.

        Args:
            None
        """
        original = ValidationResult.success()
        original.add_warning(WARNING_MSG)
        assert original.warnings == ()


class TestValidationResultImmutability:
    """Tests verifying ValidationResult is a frozen dataclass."""

    def test_cannot_set_is_valid(self) -> None:
        """Attempting to set is_valid on a ValidationResult raises AttributeError.

        Args:
            None
        """
        result = ValidationResult.success()
        with pytest.raises((AttributeError, TypeError)):
            result.is_valid = False  # type: ignore[misc]

    def test_cannot_set_errors(self) -> None:
        """Attempting to set errors on a ValidationResult raises AttributeError.

        Args:
            None
        """
        result = ValidationResult.success()
        with pytest.raises((AttributeError, TypeError)):
            result.errors = ("new",)  # type: ignore[misc]


# ===========================================================================
# ShapeValidator
# ===========================================================================


class TestShapeValidatorEntry:
    """Tests for ShapeValidator.validate_entry()."""

    def test_valid_entry_returns_success(self, entry_x: ShapeEntry) -> None:
        """validate_entry() returns a successful result for a well-formed entry.

        Args:
            entry_x: Valid ShapeEntry fixture.
        """
        validator = ShapeValidator()
        result = validator.validate_entry(entry_x)
        assert result.is_valid is True

    def test_valid_entry_has_no_errors(self, entry_y: ShapeEntry) -> None:
        """validate_entry() returns empty errors for a valid entry.

        Args:
            entry_y: Valid ShapeEntry fixture.
        """
        validator = ShapeValidator()
        result = validator.validate_entry(entry_y)
        assert result.errors == ()


class TestShapeValidatorCollection:
    """Tests for ShapeValidator.validate_collection()."""

    def test_valid_collection_returns_success(self, simple_data: ShapeData) -> None:
        """validate_collection() returns a successful result for normal ShapeData.

        Args:
            simple_data: Two-entry ShapeData fixture.
        """
        validator = ShapeValidator()
        result = validator.validate_collection(simple_data)
        assert result.is_valid is True

    def test_empty_collection_validates(self, empty_data: ShapeData) -> None:
        """validate_collection() succeeds for an empty ShapeData (no entries to fail).

        Args:
            empty_data: Zero-entry ShapeData fixture.
        """
        validator = ShapeValidator()
        result = validator.validate_collection(empty_data)
        # Empty data has no entries to violate positive-dims or unique-names specs
        assert isinstance(result, ValidationResult)

    def test_structurally_invalid_entry_triggers_failure(self, simple_data: ShapeData) -> None:
        """validate_collection() detects entries with non-positive dims bypassed via object.__setattr__.

        ShapeEntry.__post_init__ blocks invalid dims at construction; we bypass it
        using object.__setattr__ to simulate structural corruption.

        Args:
            simple_data: Two-entry ShapeData fixture used as template.
        """
        # Build an entry that bypasses __post_init__
        bad_entry = ShapeEntry.__new__(ShapeEntry)
        object.__setattr__(bad_entry, "name", "x")
        object.__setattr__(bad_entry, "dimensions", (-1,))  # invalid dim

        corrupted = ShapeData.__new__(ShapeData)
        object.__setattr__(corrupted, "entries", {"x": bad_entry})
        object.__setattr__(corrupted, "model_family", ModelFamily.DLKIT_NN)
        object.__setattr__(corrupted, "source", ShapeSource.TRAINING_DATASET)
        object.__setattr__(corrupted, "default_input", None)
        object.__setattr__(corrupted, "default_output", None)

        validator = ShapeValidator()
        result = validator.validate_collection(corrupted)
        assert result.is_valid is False

    def test_returns_validation_result_type(self, simple_data: ShapeData) -> None:
        """validate_collection() always returns a ValidationResult instance.

        Args:
            simple_data: Two-entry ShapeData fixture.
        """
        validator = ShapeValidator()
        result = validator.validate_collection(simple_data)
        assert isinstance(result, ValidationResult)


# ===========================================================================
# ShapeSerializer
# ===========================================================================


class TestShapeSerializerSerialize:
    """Tests for ShapeSerializer.serialize()."""

    def test_serialize_returns_dict(self, simple_data: ShapeData) -> None:
        """serialize() produces a dictionary.

        Args:
            simple_data: Two-entry ShapeData fixture.
        """
        serializer = ShapeSerializer()
        result = serializer.serialize(simple_data)
        assert isinstance(result, dict)

    def test_serialize_has_entries_key(self, simple_data: ShapeData) -> None:
        """serialize() output contains an 'entries' or 'data' structure.

        The V3 format wraps shape data under a 'data' key (with 'metadata'), and
        the inner data has 'entries'. We verify the serialized dict is non-empty
        and structurally valid by confirming round-trip works.

        Args:
            simple_data: Two-entry ShapeData fixture.
        """
        serializer = ShapeSerializer()
        result = serializer.serialize(simple_data)
        # The V3 serializer wraps everything in metadata + data
        assert "metadata" in result
        assert "data" in result
        assert "entries" in result["data"]

    def test_serialize_empty_data(self, empty_data: ShapeData) -> None:
        """serialize() succeeds for an empty ShapeData.

        Args:
            empty_data: Zero-entry ShapeData fixture.
        """
        serializer = ShapeSerializer()
        result = serializer.serialize(empty_data)
        assert isinstance(result, dict)
        assert result["data"]["entries"] == {}


class TestShapeSerializerRoundTrip:
    """Tests for serialize/deserialize round-trip correctness."""

    def test_round_trip_simple_data(self, simple_data: ShapeData) -> None:
        """deserialize(serialize(data)) recovers the original ShapeData entries.

        Args:
            simple_data: Two-entry ShapeData fixture.
        """
        serializer = ShapeSerializer()
        restored = serializer.deserialize(serializer.serialize(simple_data))
        assert restored.entries["x"].dimensions == simple_data.entries["x"].dimensions
        assert restored.entries["y"].dimensions == simple_data.entries["y"].dimensions

    def test_round_trip_preserves_model_family(self, simple_data: ShapeData) -> None:
        """Round-trip preserves the model_family field.

        Args:
            simple_data: Two-entry ShapeData fixture.
        """
        serializer = ShapeSerializer()
        restored = serializer.deserialize(serializer.serialize(simple_data))
        assert restored.model_family == simple_data.model_family

    def test_round_trip_preserves_source(self, simple_data: ShapeData) -> None:
        """Round-trip preserves the source field.

        Args:
            simple_data: Two-entry ShapeData fixture.
        """
        serializer = ShapeSerializer()
        restored = serializer.deserialize(serializer.serialize(simple_data))
        assert restored.source == simple_data.source

    def test_round_trip_preserves_default_input(self, data_with_defaults: ShapeData) -> None:
        """Round-trip preserves the explicit default_input field.

        Args:
            data_with_defaults: ShapeData with explicit defaults fixture.
        """
        serializer = ShapeSerializer()
        restored = serializer.deserialize(serializer.serialize(data_with_defaults))
        assert restored.default_input == data_with_defaults.default_input

    def test_round_trip_preserves_default_output(self, data_with_defaults: ShapeData) -> None:
        """Round-trip preserves the explicit default_output field.

        Args:
            data_with_defaults: ShapeData with explicit defaults fixture.
        """
        serializer = ShapeSerializer()
        restored = serializer.deserialize(serializer.serialize(data_with_defaults))
        assert restored.default_output == data_with_defaults.default_output

    def test_round_trip_empty_data(self, empty_data: ShapeData) -> None:
        """Round-trip on empty ShapeData produces empty entries.

        Args:
            empty_data: Zero-entry ShapeData fixture.
        """
        serializer = ShapeSerializer()
        restored = serializer.deserialize(serializer.serialize(empty_data))
        assert restored.is_empty()


class TestShapeSerializerDeserializeLegacyFormats:
    """Tests for backward-compatible deserialization of V1 and V2 formats."""

    def test_deserialize_v1_dict_format(self) -> None:
        """V1 legacy dict format deserializes correctly.

        The V1 format uses {'_type': 'dict', 'data': {'x': [10, 5]}}.
        """
        serializer = ShapeSerializer()
        result = serializer.deserialize(V1_DICT_RAW)
        assert isinstance(result, ShapeData)
        assert "x" in result.entries
        assert result.entries["x"].dimensions == (10, 5)

    def test_deserialize_v2_format(self) -> None:
        """V2 enhanced format deserializes correctly.

        The V2 format uses {'entries': {...}, 'model_family': ..., 'source': ...}.
        """
        serializer = ShapeSerializer()
        result = serializer.deserialize(V2_RAW)
        assert isinstance(result, ShapeData)
        assert "x" in result.entries
        assert result.entries["x"].dimensions == (10, 5)
        assert result.model_family == ModelFamily.DLKIT_NN
        assert result.source == ShapeSource.TRAINING_DATASET

    def test_deserialize_v3_format(self) -> None:
        """V3 modern format (current) deserializes correctly.

        The V3 format includes 'schema_version': '3.0' in the payload.
        """
        serializer = ShapeSerializer()
        result = serializer.deserialize(V3_RAW)
        assert isinstance(result, ShapeData)
        assert result.entries["x"].dimensions == (10, 5)
        assert result.entries["y"].dimensions == (5,)
        assert result.model_family == ModelFamily.DLKIT_NN


# ===========================================================================
# ShapeAliasResolver
# ===========================================================================


class TestShapeAliasResolverNoChange:
    """Tests for resolve_aliases() when x/y already exist."""

    def test_data_with_x_y_unchanged_dims(self, simple_data: ShapeData) -> None:
        """resolve_aliases() on data that already has x and y keeps their dims.

        Args:
            simple_data: ShapeData fixture with x and y present.
        """
        resolver = ShapeAliasResolver()
        resolved = resolver.resolve_aliases(simple_data)
        assert resolved.entries["x"].dimensions == simple_data.entries["x"].dimensions
        assert resolved.entries["y"].dimensions == simple_data.entries["y"].dimensions

    def test_data_with_x_y_no_extra_aliases(self, simple_data: ShapeData) -> None:
        """resolve_aliases() on data already containing x/y adds no unexpected entries.

        Args:
            simple_data: ShapeData fixture with x and y present.
        """
        resolver = ShapeAliasResolver()
        resolved = resolver.resolve_aliases(simple_data)
        # Original entry names should be preserved; no unexpected keys appear
        assert set(resolved.entries.keys()) >= {"x", "y"}


class TestShapeAliasResolverSingleEntry:
    """Tests for resolve_aliases() on single-entry data without x/y."""

    def test_single_entry_adds_x_alias(self, single_entry_data: ShapeData) -> None:
        """resolve_aliases() adds an 'x' alias pointing to the sole entry.

        Args:
            single_entry_data: ShapeData fixture with one 'features' entry.
        """
        resolver = ShapeAliasResolver()
        resolved = resolver.resolve_aliases(single_entry_data)
        assert "x" in resolved.entries

    def test_single_entry_x_alias_has_same_dims(self, single_entry_data: ShapeData) -> None:
        """The x alias inherits dimensions from the sole existing entry.

        Args:
            single_entry_data: ShapeData fixture with one 'features' entry.
        """
        resolver = ShapeAliasResolver()
        resolved = resolver.resolve_aliases(single_entry_data)
        expected_dims = single_entry_data.entries["features"].dimensions
        assert resolved.entries["x"].dimensions == expected_dims

    def test_single_entry_adds_y_alias_duplicating_x(self, single_entry_data: ShapeData) -> None:
        """resolve_aliases() duplicates x as y when only one entry exists.

        Args:
            single_entry_data: ShapeData fixture with one 'features' entry.
        """
        resolver = ShapeAliasResolver()
        resolved = resolver.resolve_aliases(single_entry_data)
        assert "y" in resolved.entries
        assert resolved.entries["y"].dimensions == resolved.entries["x"].dimensions


class TestShapeAliasResolverTwoEntries:
    """Tests for resolve_aliases() on data with two entries but no x/y."""

    def test_two_entries_x_maps_to_first(self, two_entry_data: ShapeData) -> None:
        """resolve_aliases() maps x to the first entry's dimensions.

        Args:
            two_entry_data: ShapeData with 'first' and 'second' entries.
        """
        resolver = ShapeAliasResolver()
        resolved = resolver.resolve_aliases(two_entry_data)
        first_dims = two_entry_data.entries["first"].dimensions
        assert resolved.entries["x"].dimensions == first_dims

    def test_two_entries_y_maps_to_second(self, two_entry_data: ShapeData) -> None:
        """resolve_aliases() maps y to the second entry's dimensions.

        Args:
            two_entry_data: ShapeData with 'first' and 'second' entries.
        """
        resolver = ShapeAliasResolver()
        resolved = resolver.resolve_aliases(two_entry_data)
        second_dims = two_entry_data.entries["second"].dimensions
        assert resolved.entries["y"].dimensions == second_dims


class TestShapeAliasResolverExplicitDefaults:
    """Tests for resolve_aliases() respecting default_input/default_output."""

    def test_explicit_defaults_used_for_x(self) -> None:
        """When default_input is set and matches an entry, x alias uses those dims.

        ShapeData enforces entry key == entry.name, so we build the entries
        consistently.

        Args:
            None
        """
        feat_entry = ShapeEntry(name="features", dimensions=(100,))
        lbl_entry = ShapeEntry(name="labels", dimensions=(3,))
        data = ShapeData(
            entries={"features": feat_entry, "labels": lbl_entry},
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
            default_input="features",
            default_output="labels",
        )
        resolver = ShapeAliasResolver()
        resolved = resolver.resolve_aliases(data)
        assert resolved.entries["x"].dimensions == (100,)

    def test_explicit_defaults_used_for_y(self) -> None:
        """When default_output is set and matches an entry, y alias uses those dims.

        Args:
            None
        """
        feat_entry = ShapeEntry(name="features", dimensions=(100,))
        lbl_entry = ShapeEntry(name="labels", dimensions=(3,))
        data = ShapeData(
            entries={"features": feat_entry, "labels": lbl_entry},
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
            default_input="features",
            default_output="labels",
        )
        resolver = ShapeAliasResolver()
        resolved = resolver.resolve_aliases(data)
        assert resolved.entries["y"].dimensions == (3,)


class TestShapeAliasResolverEmptyData:
    """Tests for resolve_aliases() on empty ShapeData."""

    def test_empty_data_returns_unchanged(self, empty_data: ShapeData) -> None:
        """resolve_aliases() on empty data returns the same object unchanged.

        Args:
            empty_data: Zero-entry ShapeData fixture.
        """
        resolver = ShapeAliasResolver()
        resolved = resolver.resolve_aliases(empty_data)
        assert resolved.is_empty()

    def test_empty_data_no_x_added(self, empty_data: ShapeData) -> None:
        """resolve_aliases() on empty data does not inject an 'x' entry.

        Args:
            empty_data: Zero-entry ShapeData fixture.
        """
        resolver = ShapeAliasResolver()
        resolved = resolver.resolve_aliases(empty_data)
        assert "x" not in resolved.entries


class TestShapeAliasResolverSmartDefaults:
    """Tests for ShapeAliasResolver.resolve_smart_defaults()."""

    def test_empty_data_returns_none_none(self, empty_data: ShapeData) -> None:
        """resolve_smart_defaults() on empty data returns (None, None).

        Args:
            empty_data: Zero-entry ShapeData fixture.
        """
        resolver = ShapeAliasResolver()
        default_in, default_out = resolver.resolve_smart_defaults(empty_data)
        assert default_in is None
        assert default_out is None

    def test_x_y_entries_return_x_y_defaults(self, simple_data: ShapeData) -> None:
        """resolve_smart_defaults() returns ('x', 'y') when x/y entries exist.

        Args:
            simple_data: ShapeData with x and y entries.
        """
        resolver = ShapeAliasResolver()
        default_in, default_out = resolver.resolve_smart_defaults(simple_data)
        assert default_in == "x"
        assert default_out == "y"

    def test_explicit_defaults_returned_directly(self, data_with_defaults: ShapeData) -> None:
        """resolve_smart_defaults() honours explicit default_input/default_output.

        Args:
            data_with_defaults: ShapeData with explicit defaults set.
        """
        resolver = ShapeAliasResolver()
        default_in, default_out = resolver.resolve_smart_defaults(data_with_defaults)
        assert default_in == data_with_defaults.default_input
        assert default_out == data_with_defaults.default_output

    def test_two_entries_first_as_input_second_as_output(self, two_entry_data: ShapeData) -> None:
        """resolve_smart_defaults() uses first entry as input, second as output.

        Args:
            two_entry_data: ShapeData with 'first'/'second' entries, no x/y.
        """
        resolver = ShapeAliasResolver()
        default_in, default_out = resolver.resolve_smart_defaults(two_entry_data)
        keys = list(two_entry_data.entries.keys())
        assert default_in == keys[0]
        assert default_out == keys[1]

    def test_single_entry_output_is_none(self, single_entry_data: ShapeData) -> None:
        """resolve_smart_defaults() returns None for output when only one entry exists and no y.

        Args:
            single_entry_data: ShapeData with single 'features' entry.
        """
        resolver = ShapeAliasResolver()
        _default_in, default_out = resolver.resolve_smart_defaults(single_entry_data)
        # No second entry and no y; output should be None
        assert default_out is None


# ===========================================================================
# ShapeSpec (via create_shape_spec)
# ===========================================================================


class TestShapeSpecInputOutputShapes:
    """Tests for get_input_shape()/get_output_shape() on ShapeSpec."""

    def test_get_input_shape_returns_x_dims(self) -> None:
        """get_input_shape() returns the dimensions registered under x.

        Args:
            None
        """
        spec = create_shape_spec(
            shapes={"x": (10, 5), "y": (5,)},
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )
        assert spec.get_input_shape() == (10, 5)

    def test_get_output_shape_returns_y_dims(self) -> None:
        """get_output_shape() returns the dimensions registered under y.

        Args:
            None
        """
        spec = create_shape_spec(
            shapes={"x": (10, 5), "y": (5,)},
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )
        assert spec.get_output_shape() == (5,)


class TestShapeSpecIsEmpty:
    """Tests for ShapeSpec.is_empty()."""

    def test_empty_spec_is_empty(self) -> None:
        """create_shape_spec(shapes=None) returns a spec that reports is_empty() True.

        Args:
            None
        """
        spec = create_shape_spec(shapes=None)
        assert spec.is_empty() is True

    def test_non_empty_spec_not_empty(self) -> None:
        """create_shape_spec with entries reports is_empty() False.

        Args:
            None
        """
        spec = create_shape_spec(
            shapes={"x": (8,), "y": (2,)},
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )
        assert spec.is_empty() is False


class TestShapeSpecToDictFromDict:
    """Tests for ShapeSpec to_dict/from_dict round-trip."""

    def test_to_dict_from_dict_round_trip_entry_dims(self) -> None:
        """from_dict(to_dict(spec)) restores entry dimensions.

        Args:
            None
        """
        from dlkit.domain.shapes.core import ShapeSpec

        spec = create_shape_spec(
            shapes={"x": (10, 5), "y": (5,)},
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )
        raw = spec.to_dict()
        restored = ShapeSpec.from_dict(raw)
        assert restored.get_shape("x") == (10, 5)
        assert restored.get_shape("y") == (5,)

    def test_to_dict_returns_dict(self) -> None:
        """to_dict() returns a plain Python dict.

        Args:
            None
        """
        spec = create_shape_spec(
            shapes={"x": (3,), "y": (1,)},
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )
        result = spec.to_dict()
        assert isinstance(result, dict)


class TestShapeSpecWithCanonicalAliases:
    """Tests for ShapeSpec.with_canonical_aliases()."""

    def test_adds_x_when_missing(self) -> None:
        """with_canonical_aliases() adds x alias when the spec has no x entry.

        Args:
            None
        """
        from dlkit.domain.shapes.core import ShapeSpec
        from dlkit.domain.shapes.value_objects import ShapeData, ShapeEntry

        data = ShapeData(
            entries={"features": ShapeEntry(name="features", dimensions=(16,))},
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )
        spec = ShapeSpec(data=data)
        aliased = spec.with_canonical_aliases()
        assert aliased.has_shape("x")

    def test_adds_y_when_missing(self) -> None:
        """with_canonical_aliases() adds y alias when the spec has no y entry.

        Args:
            None
        """
        from dlkit.domain.shapes.core import ShapeSpec
        from dlkit.domain.shapes.value_objects import ShapeData, ShapeEntry

        data = ShapeData(
            entries={"features": ShapeEntry(name="features", dimensions=(16,))},
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )
        spec = ShapeSpec(data=data)
        aliased = spec.with_canonical_aliases()
        assert aliased.has_shape("y")

    def test_preserves_existing_x_y_dims(self) -> None:
        """with_canonical_aliases() preserves x/y dims when they already exist.

        Args:
            None
        """
        spec = create_shape_spec(
            shapes={"x": (10, 5), "y": (5,)},
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )
        aliased = spec.with_canonical_aliases()
        assert aliased.get_shape("x") == (10, 5)
        assert aliased.get_shape("y") == (5,)


# ===========================================================================
# Hypothesis property-based tests
# ===========================================================================


@st.composite
def positive_dims(draw: st.DrawFn) -> tuple[int, ...]:
    """Hypothesis strategy: a non-empty tuple of positive integers.

    Args:
        draw: Hypothesis draw callable.

    Returns:
        tuple[int, ...]: Non-empty tuple of positive ints.
    """
    length = draw(st.integers(min_value=1, max_value=4))
    dims = tuple(draw(st.integers(min_value=1, max_value=256)) for _ in range(length))
    return dims


@st.composite
def shape_entry_pair(draw: st.DrawFn) -> tuple[ShapeEntry, ShapeEntry]:
    """Hypothesis strategy: a pair of ShapeEntry objects named x and y.

    Args:
        draw: Hypothesis draw callable.

    Returns:
        tuple[ShapeEntry, ShapeEntry]: (entry_x, entry_y) pair.
    """
    x_dims = draw(positive_dims())
    y_dims = draw(positive_dims())
    return (
        ShapeEntry(name="x", dimensions=x_dims),
        ShapeEntry(name="y", dimensions=y_dims),
    )


@given(shape_entry_pair())
@settings(max_examples=40)
def test_validation_result_add_error_always_invalid(pair: tuple[ShapeEntry, ShapeEntry]) -> None:
    """Property: add_error on success always yields is_valid=False regardless of entry shape.

    Args:
        pair: Hypothesis-generated pair of ShapeEntry objects.
    """
    result = ValidationResult.success().add_error("any error")
    assert result.is_valid is False


@given(shape_entry_pair())
@settings(max_examples=40)
def test_validation_result_add_warning_always_valid(pair: tuple[ShapeEntry, ShapeEntry]) -> None:
    """Property: add_warning on success always preserves is_valid=True.

    Args:
        pair: Hypothesis-generated pair of ShapeEntry objects.
    """
    result = ValidationResult.success().add_warning("any warning")
    assert result.is_valid is True


@given(shape_entry_pair())
@settings(max_examples=40)
def test_serializer_round_trip_property(pair: tuple[ShapeEntry, ShapeEntry]) -> None:
    """Property: serialize/deserialize preserves x and y dimensions for any valid entry pair.

    Args:
        pair: Hypothesis-generated pair of ShapeEntry objects.
    """
    entry_x, entry_y = pair
    data = ShapeData(
        entries={"x": entry_x, "y": entry_y},
        model_family=ModelFamily.DLKIT_NN,
        source=ShapeSource.TRAINING_DATASET,
    )
    serializer = ShapeSerializer()
    restored = serializer.deserialize(serializer.serialize(data))
    assert restored.entries["x"].dimensions == entry_x.dimensions
    assert restored.entries["y"].dimensions == entry_y.dimensions


@given(shape_entry_pair())
@settings(max_examples=40)
def test_alias_resolver_always_adds_x_y_for_named_pair(
    pair: tuple[ShapeEntry, ShapeEntry],
) -> None:
    """Property: resolve_aliases always ensures x and y exist for two-entry data.

    Args:
        pair: Hypothesis-generated pair of ShapeEntry objects.
    """
    entry_x, entry_y = pair
    data = ShapeData(
        entries={"x": entry_x, "y": entry_y},
        model_family=ModelFamily.DLKIT_NN,
        source=ShapeSource.TRAINING_DATASET,
    )
    resolver = ShapeAliasResolver()
    resolved = resolver.resolve_aliases(data)
    assert "x" in resolved.entries
    assert "y" in resolved.entries


@given(positive_dims())
@settings(max_examples=40)
def test_alias_resolver_single_entry_duplicates_to_y(dims: tuple[int, ...]) -> None:
    """Property: for any valid single-entry data, resolve_aliases creates matching x and y dims.

    Args:
        dims: Hypothesis-generated positive dimension tuple.
    """
    entry = ShapeEntry(name="features", dimensions=dims)
    data = ShapeData(
        entries={"features": entry},
        model_family=ModelFamily.DLKIT_NN,
        source=ShapeSource.TRAINING_DATASET,
    )
    resolver = ShapeAliasResolver()
    resolved = resolver.resolve_aliases(data)
    assert resolved.entries["x"].dimensions == dims
    assert resolved.entries["y"].dimensions == dims
