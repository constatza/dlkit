"""Tests for the pure patching functions in dlkit.tools.config.core.patching.

All test data is created via fixtures.  No inline data construction inside
test functions (except for expected values in assert statements).

Coverage:
- split_overrides
- insert_path
- compile_dotted_overrides
- strict_merge_patches
- compile_mixed_overrides
- iter_validated_updates / apply_patch
- patch_model (end-to-end)
- exclude=True field round-trip (critical correctness guarantee)
- ComponentSettings extra="allow" passthrough
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dlkit.tools.config.core.patching import (
    apply_patch,
    compile_dotted_overrides,
    compile_mixed_overrides,
    insert_path,
    iter_validated_updates,
    patch_model,
    split_overrides,
    strict_merge_patches,
)


# ---------------------------------------------------------------------------
# Minimal test models
# ---------------------------------------------------------------------------


class FlatModel(BaseModel):
    """Simple flat model with scalar fields."""

    model_config = ConfigDict(frozen=True)

    name: str
    count: int
    ratio: float = 1.0


class InnerModel(BaseModel):
    """Nested model used inside OuterModel."""

    model_config = ConfigDict(frozen=True)

    x: int
    y: int = 0


class OuterModel(BaseModel):
    """Model containing a nested BaseModel field."""

    model_config = ConfigDict(frozen=True)

    label: str
    inner: InnerModel


class ExtraModel(BaseModel):
    """Model allowing extra fields (mirrors ComponentSettings behaviour)."""

    model_config = ConfigDict(extra="allow")

    name: str


class ExcludedFieldModel(BaseModel):
    """Model with an excluded field — mirrors ValueFeature.value semantics."""

    model_config = ConfigDict(frozen=True)

    label: str
    secret: Any = Field(default=None, exclude=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_model() -> FlatModel:
    """FlatModel with default ratio."""
    return FlatModel(name="alpha", count=10)


@pytest.fixture
def outer_model() -> OuterModel:
    """OuterModel with nested InnerModel."""
    return OuterModel(label="top", inner=InnerModel(x=1, y=2))


@pytest.fixture
def extra_model() -> ExtraModel:
    """ExtraModel with one extra field pre-set."""
    return ExtraModel(name="comp", extra_field="hello")


@pytest.fixture
def excluded_model_with_array() -> ExcludedFieldModel:
    """ExcludedFieldModel with a numpy array in the excluded field."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    return ExcludedFieldModel(label="data", secret=arr)


@pytest.fixture
def excluded_model_with_object() -> ExcludedFieldModel:
    """ExcludedFieldModel with an arbitrary Python object in the excluded field."""
    return ExcludedFieldModel(label="obj", secret={"nested": True})


# ---------------------------------------------------------------------------
# TestSplitOverrides
# ---------------------------------------------------------------------------


class TestSplitOverrides:
    """Tests for split_overrides()."""

    @pytest.mark.parametrize(
        "overrides, expected_dotted_keys, expected_nested_keys",
        [
            ({}, [], []),
            ({"a": 1}, [], ["a"]),
            ({"a.b": 1}, ["a.b"], []),
            ({"a.b": 1, "c": 2}, ["a.b"], ["c"]),
            ({"a.b.c": 1, "d.e": 2, "f": 3}, ["a.b.c", "d.e"], ["f"]),
        ],
        ids=[
            "empty",
            "plain-only",
            "dotted-only",
            "mixed",
            "multi-dotted-plus-plain",
        ],
    )
    def test_bucket_keys(
        self,
        overrides: dict,
        expected_dotted_keys: list,
        expected_nested_keys: list,
    ) -> None:
        dotted, nested = split_overrides(overrides)
        assert set(dotted.keys()) == set(expected_dotted_keys)
        assert set(nested.keys()) == set(expected_nested_keys)

    def test_values_preserved_in_each_bucket(self) -> None:
        overrides = {"a.b": 42, "c": "hello"}
        dotted, nested = split_overrides(overrides)
        assert dotted["a.b"] == 42
        assert nested["c"] == "hello"

    def test_custom_separator(self) -> None:
        overrides = {"a/b": 1, "c": 2}
        dotted, nested = split_overrides(overrides, sep="/")
        assert "a/b" in dotted
        assert "c" in nested

    def test_empty_returns_two_empty_dicts(self) -> None:
        d, n = split_overrides({})
        assert d == {} and n == {}


# ---------------------------------------------------------------------------
# TestInsertPath
# ---------------------------------------------------------------------------


class TestInsertPath:
    """Tests for insert_path()."""

    @pytest.mark.parametrize(
        "key, value, expected_root",
        [
            ("a", 1, {"a": 1}),
            ("a.b", 1, {"a": {"b": 1}}),
            ("a.b.c", 1, {"a": {"b": {"c": 1}}}),
        ],
        ids=["single-segment", "two-segments", "three-segments"],
    )
    def test_happy_path(self, key: str, value: Any, expected_root: dict) -> None:
        root: dict[str, Any] = {}
        result = insert_path(root, key=key, value=value)
        assert result == expected_root
        assert result is root  # returns same dict

    def test_shared_prefix_builds_subtree(self) -> None:
        root: dict[str, Any] = {}
        insert_path(root, key="a.b", value=1)
        insert_path(root, key="a.c", value=2)
        assert root == {"a": {"b": 1, "c": 2}}

    @pytest.mark.parametrize(
        "key, exc_fragment",
        [
            ("", "non-empty"),
            ("a..b", "empty segment"),
            (".a", "empty segment"),
        ],
        ids=["empty-key", "double-dot", "leading-dot"],
    )
    def test_invalid_key_raises(self, key: str, exc_fragment: str) -> None:
        with pytest.raises(ValueError, match=exc_fragment):
            insert_path({}, key=key, value=1)

    def test_traverse_into_leaf_raises(self) -> None:
        root: dict[str, Any] = {"a": 99}
        with pytest.raises(ValueError, match="leaf"):
            insert_path(root, key="a.b", value=1)

    def test_overwrite_subtree_with_leaf_raises(self) -> None:
        root: dict[str, Any] = {}
        insert_path(root, key="a.b", value=1)
        with pytest.raises(ValueError, match="parent"):
            insert_path(root, key="a", value=2)

    def test_empty_sep_raises(self) -> None:
        with pytest.raises(ValueError, match="sep"):
            insert_path({}, key="a.b", value=1, sep="")


# ---------------------------------------------------------------------------
# TestCompileDottedOverrides
# ---------------------------------------------------------------------------


class TestCompileDottedOverrides:
    """Tests for compile_dotted_overrides()."""

    @pytest.mark.parametrize(
        "dotted, expected",
        [
            ({}, {}),
            ({"a.b": 1}, {"a": {"b": 1}}),
            ({"a.b": 1, "a.c": 2}, {"a": {"b": 1, "c": 2}}),
            ({"a.b.c": 1, "a.b.d": 2}, {"a": {"b": {"c": 1, "d": 2}}}),
            ({"x.y": "val"}, {"x": {"y": "val"}}),
        ],
        ids=["empty", "single", "shared-prefix", "deep-shared", "string-value"],
    )
    def test_expansion(self, dotted: dict, expected: dict) -> None:
        assert compile_dotted_overrides(dotted) == expected

    def test_non_string_key_raises(self) -> None:
        with pytest.raises(ValueError, match="strings"):
            compile_dotted_overrides({1: "oops"})  # type: ignore[dict-item]

    def test_collision_raises(self) -> None:
        # "a.b.c" inserts {"a": {"b": {"c": 1}}}; then "a.b" tries to set
        # "b" to 2 but "b" is already a subtree → structural conflict.
        # (Using two distinct keys avoids Python's silent dict-key deduplication.)
        with pytest.raises(ValueError):
            compile_dotted_overrides({"a.b.c": 1, "a.b": 2})


# ---------------------------------------------------------------------------
# TestStrictMergePatches
# ---------------------------------------------------------------------------


class TestStrictMergePatches:
    """Tests for strict_merge_patches()."""

    @pytest.mark.parametrize(
        "left, right, expected",
        [
            ({}, {}, {}),
            ({"a": 1}, {}, {"a": 1}),
            ({}, {"b": 2}, {"b": 2}),
            ({"a": 1}, {"b": 2}, {"a": 1, "b": 2}),
            ({"a": {"x": 1}}, {"a": {"y": 2}}, {"a": {"x": 1, "y": 2}}),
        ],
        ids=["both-empty", "left-only", "right-only", "disjoint", "nested-merge"],
    )
    def test_clean_merges(self, left: dict, right: dict, expected: dict) -> None:
        assert strict_merge_patches(left, right) == expected

    @pytest.mark.parametrize(
        "left, right, desc",
        [
            ({"a": 1}, {"a": 2}, "leaf-leaf collision"),
            ({"a": {"b": 1}}, {"a": 2}, "dict-leaf collision"),
            ({"a": 2}, {"a": {"b": 1}}, "leaf-dict collision"),
        ],
        ids=["leaf-leaf", "dict-leaf", "leaf-dict"],
    )
    def test_collision_raises(self, left: dict, right: dict, desc: str) -> None:
        with pytest.raises(ValueError, match="collision|Conflict"):
            strict_merge_patches(left, right)

    def test_returns_new_dict(self) -> None:
        left = {"a": 1}
        right = {"b": 2}
        result = strict_merge_patches(left, right)
        assert result is not left
        assert result is not right


# ---------------------------------------------------------------------------
# TestCompileMixedOverrides
# ---------------------------------------------------------------------------


class TestCompileMixedOverrides:
    """Tests for compile_mixed_overrides()."""

    @pytest.mark.parametrize(
        "overrides, expected",
        [
            ({}, {}),
            ({"a": 1}, {"a": 1}),
            ({"a.b": 1}, {"a": {"b": 1}}),
            ({"a.b": 1, "c": 2}, {"a": {"b": 1}, "c": 2}),
            ({"a.b": 1, "d": {"e": 3}}, {"a": {"b": 1}, "d": {"e": 3}}),
        ],
        ids=["empty", "plain-only", "dotted-only", "mixed-flat", "mixed-nested"],
    )
    def test_compilation(self, overrides: dict, expected: dict) -> None:
        assert compile_mixed_overrides(overrides) == expected

    def test_collision_between_dotted_and_nested_raises(self) -> None:
        # "a.b" compiled → {"a": {"b": 1}}; nested "a" → {"a": 2}
        # Both target key "a" → collision
        with pytest.raises(ValueError):
            compile_mixed_overrides({"a.b": 1, "a": 2})

    def test_custom_sep(self) -> None:
        result = compile_mixed_overrides({"a/b": 1, "c": 2}, sep="/")
        assert result == {"a": {"b": 1}, "c": 2}


# ---------------------------------------------------------------------------
# TestApplyPatch — flat and nested models
# ---------------------------------------------------------------------------


class TestApplyPatch:
    """Tests for apply_patch() on plain frozen BaseModel instances."""

    def test_flat_field_replacement(self, flat_model: FlatModel) -> None:
        result = apply_patch(flat_model, {"name": "beta"})
        assert result.name == "beta"
        assert result.count == flat_model.count  # unchanged

    def test_multiple_flat_fields(self, flat_model: FlatModel) -> None:
        result = apply_patch(flat_model, {"name": "gamma", "count": 99})
        assert result.name == "gamma"
        assert result.count == 99

    def test_returns_new_instance(self, flat_model: FlatModel) -> None:
        result = apply_patch(flat_model, {"name": "delta"})
        assert result is not flat_model

    def test_empty_patch_returns_deep_copy(self, flat_model: FlatModel) -> None:
        result = apply_patch(flat_model, {})
        assert result == flat_model
        assert result is not flat_model

    def test_nested_model_recurse(self, outer_model: OuterModel) -> None:
        result = apply_patch(outer_model, {"inner": {"x": 99}})
        assert result.inner.x == 99
        assert result.inner.y == outer_model.inner.y  # preserved

    def test_unknown_field_raises_key_error(self, flat_model: FlatModel) -> None:
        with pytest.raises(KeyError, match="no_such_field"):
            apply_patch(flat_model, {"no_such_field": "oops"})

    def test_type_mismatch_raises_validation_error(self, flat_model: FlatModel) -> None:
        with pytest.raises(ValidationError):
            apply_patch(flat_model, {"count": "not-an-int"})

    @pytest.mark.parametrize(
        "revalidate",
        [True, False],
        ids=["revalidate-on", "revalidate-off"],
    )
    def test_revalidate_flag(self, flat_model: FlatModel, revalidate: bool) -> None:
        result = apply_patch(flat_model, {"name": "test"}, revalidate=revalidate)
        assert result.name == "test"

    def test_label_unchanged_on_nested_patch(self, outer_model: OuterModel) -> None:
        result = apply_patch(outer_model, {"inner": {"y": 77}})
        assert result.label == outer_model.label
        assert result.inner.y == 77


# ---------------------------------------------------------------------------
# TestExcludedFieldRoundTrip  — THE CRITICAL TEST
# ---------------------------------------------------------------------------


class TestExcludedFieldRoundTrip:
    """Verify that exclude=True fields survive apply_patch() with revalidate=True.

    This is the primary correctness guarantee for the immutable migration.
    If this test fails, the migration strategy must be revised.
    """

    def test_numpy_array_survives_revalidate(
        self, excluded_model_with_array: ExcludedFieldModel
    ) -> None:
        original_arr = excluded_model_with_array.secret
        result = apply_patch(excluded_model_with_array, {"label": "updated"}, revalidate=True)
        assert result.label == "updated"
        assert result.secret is not None
        np.testing.assert_array_equal(result.secret, original_arr)

    def test_numpy_array_survives_no_revalidate(
        self, excluded_model_with_array: ExcludedFieldModel
    ) -> None:
        original_arr = excluded_model_with_array.secret
        result = apply_patch(excluded_model_with_array, {"label": "updated"}, revalidate=False)
        assert result.label == "updated"
        np.testing.assert_array_equal(result.secret, original_arr)

    def test_object_survives_revalidate(
        self, excluded_model_with_object: ExcludedFieldModel
    ) -> None:
        original_obj = excluded_model_with_object.secret
        result = apply_patch(excluded_model_with_object, {"label": "patched"}, revalidate=True)
        assert result.label == "patched"
        assert result.secret == original_obj

    def test_excluded_field_can_be_patched(
        self, excluded_model_with_array: ExcludedFieldModel
    ) -> None:
        # exclude=True only affects serialisation — the field IS in model_fields.
        # Patching it replaces the value normally.
        new_arr = np.zeros(3, dtype=np.float32)
        result = apply_patch(excluded_model_with_array, {"secret": new_arr})
        np.testing.assert_array_equal(result.secret, new_arr)

    def test_empty_patch_preserves_excluded_field(
        self, excluded_model_with_array: ExcludedFieldModel
    ) -> None:
        original_arr = excluded_model_with_array.secret
        result = apply_patch(excluded_model_with_array, {})
        np.testing.assert_array_equal(result.secret, original_arr)

    def test_value_feature_excluded_field_survives(self) -> None:
        """End-to-end test using actual ValueFeature from dlkit."""
        from dlkit.tools.config.data_entries import ValueFeature

        arr = np.ones((10, 5), dtype=np.float32)
        feature = ValueFeature(name="x", value=arr)

        # Patch an unrelated attribute (name).
        # The value field (exclude=True) must survive.
        result = apply_patch(feature, {"name": "y"}, revalidate=True)

        assert result.name == "y"
        assert result.value is not None
        np.testing.assert_array_equal(result.value, arr)


# ---------------------------------------------------------------------------
# TestExtraFieldPassthrough
# ---------------------------------------------------------------------------


class TestExtraFieldPassthrough:
    """Verify that extra="allow" fields pass through iter_validated_updates."""

    def test_extra_field_patched(self, extra_model: ExtraModel) -> None:
        result = apply_patch(extra_model, {"extra_field": "world"})
        assert result.extra_field == "world"  # type: ignore[attr-defined]

    def test_new_extra_field_added(self, extra_model: ExtraModel) -> None:
        result = apply_patch(extra_model, {"brand_new": 42})
        assert result.brand_new == 42  # type: ignore[attr-defined]

    def test_declared_field_still_validated(self, extra_model: ExtraModel) -> None:
        # "name" is a declared field — validation still applies
        with pytest.raises(ValidationError):
            apply_patch(extra_model, {"name": 999})  # int, not str


# ---------------------------------------------------------------------------
# TestPatchModel — public entrypoint
# ---------------------------------------------------------------------------


class TestPatchModel:
    """End-to-end tests for patch_model()."""

    def test_flat_override(self, flat_model: FlatModel) -> None:
        result = patch_model(flat_model, {"count": 55})
        assert result.count == 55
        assert result.name == flat_model.name

    def test_dotted_key_expands(self, outer_model: OuterModel) -> None:
        result = patch_model(outer_model, {"inner.x": 100})
        assert result.inner.x == 100
        assert result.inner.y == outer_model.inner.y

    def test_mixed_overrides(self, outer_model: OuterModel) -> None:
        result = patch_model(outer_model, {"label": "new", "inner.x": 7})
        assert result.label == "new"
        assert result.inner.x == 7

    def test_empty_overrides_returns_copy(self, flat_model: FlatModel) -> None:
        result = patch_model(flat_model, {})
        assert result == flat_model
        assert result is not flat_model

    def test_collision_raises(self, outer_model: OuterModel) -> None:
        with pytest.raises(ValueError):
            patch_model(outer_model, {"inner.x": 1, "inner": {"x": 2}})

    def test_custom_sep(self, outer_model: OuterModel) -> None:
        result = patch_model(outer_model, {"inner/x": 5}, sep="/")
        assert result.inner.x == 5

    def test_type_error_raises(self, flat_model: FlatModel) -> None:
        with pytest.raises(ValidationError):
            patch_model(flat_model, {"ratio": "not-a-float"})

    def test_unknown_key_raises(self, flat_model: FlatModel) -> None:
        with pytest.raises(KeyError):
            patch_model(flat_model, {"ghost": 1})

    @pytest.mark.parametrize(
        "overrides, expected_count, expected_name",
        [
            ({"count": 1}, 1, "alpha"),
            ({"count": 2, "name": "z"}, 2, "z"),
        ],
        ids=["single-field", "two-fields"],
    )
    def test_parametrized_flat_patches(
        self,
        flat_model: FlatModel,
        overrides: dict,
        expected_count: int,
        expected_name: str,
    ) -> None:
        result = patch_model(flat_model, overrides)
        assert result.count == expected_count
        assert result.name == expected_name
