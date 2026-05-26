"""Tests for DataEntry fields: model_input, field_role, and geometry_kind.

Covers:
- model_input bool field: True/False/non-bool rejection
- field_role: parses from string config values, defaults to FEATURE
- geometry_kind: parses from string config values, defaults to TABULAR
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from dlkit.common.geometry import FieldRole, GeometryKind
from dlkit.infrastructure.config.data_entries import Feature, PathFeature, ValueFeature


class TestModelInputField:
    """Tests for the model_input bool field on DataEntry subclasses."""

    def test_model_input_true_accepted(self) -> None:
        """True is accepted (include as model input)."""
        feat = Feature("x", value=torch.zeros(4, 3), model_input=True)
        assert feat.model_input is True

    def test_model_input_false_accepted(self) -> None:
        """False is accepted (exclude from model call)."""
        feat = Feature("x", value=torch.zeros(4, 3), model_input=False)
        assert feat.model_input is False

    def test_model_input_default_is_true(self) -> None:
        """Default model_input is True (include as model input)."""
        feat = Feature("x", value=torch.zeros(4, 3))
        assert feat.model_input is True

    def test_model_input_str_raises(self) -> None:
        """String model_input raises ValidationError (hard-cut to bool)."""
        with pytest.raises(ValidationError):
            Feature("x", value=torch.zeros(4, 3), model_input="hidden")  # type: ignore[arg-type]

    def test_model_input_int_raises(self) -> None:
        """Integer model_input raises ValidationError (hard-cut to bool)."""
        with pytest.raises(ValidationError):
            Feature("x", value=torch.zeros(4, 3), model_input=0)  # type: ignore[arg-type]

    def test_model_input_none_raises(self) -> None:
        """None model_input raises ValidationError (hard-cut to bool)."""
        with pytest.raises(ValidationError):
            Feature("x", value=torch.zeros(4, 3), model_input=None)  # type: ignore[arg-type]

    def test_model_input_int_one_raises(self) -> None:
        """Integer 1 raises ValidationError (strict bool — no coercion from int)."""
        with pytest.raises(ValidationError):
            Feature("x", value=torch.zeros(4, 3), model_input=1)  # type: ignore[arg-type]

    def test_model_input_kwarg_str_raises(self) -> None:
        """String 'kwarg' raises ValidationError (strict bool — no strings accepted)."""
        with pytest.raises(ValidationError):
            Feature("x", value=torch.zeros(4, 3), model_input="kwarg")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fixtures for field_role / geometry_kind tests
# ---------------------------------------------------------------------------


@pytest.fixture
def value_feature_default() -> ValueFeature:
    """ValueFeature constructed with no explicit field_role or geometry_kind."""
    return ValueFeature(name="x", value=torch.zeros(4, 3))


@pytest.fixture
def path_feature_placeholder() -> PathFeature:
    """PathFeature placeholder with no path (no data source)."""
    return PathFeature(name="x")


# ---------------------------------------------------------------------------
# field_role tests
# ---------------------------------------------------------------------------


class TestFieldRoleField:
    """DataEntry.field_role parses correctly from FieldRole enum values."""

    def test_default_field_role_is_feature(self, value_feature_default: ValueFeature) -> None:
        """Default field_role is FieldRole.FEATURE."""
        assert value_feature_default.field_role == FieldRole.FEATURE

    @pytest.mark.parametrize("role", list(FieldRole))
    def test_field_role_set_from_enum(self, role: FieldRole) -> None:
        """field_role accepts any FieldRole enum value."""
        feat = ValueFeature(name="x", value=torch.zeros(4, 3), field_role=role)
        assert feat.field_role == role

    @pytest.mark.parametrize(
        "role_str, expected",
        [
            ("feature", FieldRole.FEATURE),
            ("feature_coordinates", FieldRole.FEATURE_COORDINATES),
            ("target_coordinates", FieldRole.TARGET_COORDINATES),
        ],
    )
    def test_field_role_parses_from_string(self, role_str: str, expected: FieldRole) -> None:
        """field_role accepts the string value defined by FieldRole (StrEnum)."""
        feat = ValueFeature(name="x", value=torch.zeros(4, 3), field_role=role_str)  # type: ignore[arg-type]
        assert feat.field_role == expected

    def test_field_role_excluded_from_model_dump(self, value_feature_default: ValueFeature) -> None:
        """field_role does not appear in model_dump() output (exclude=True)."""
        dumped = value_feature_default.model_dump()
        assert "field_role" not in dumped


# ---------------------------------------------------------------------------
# geometry_kind tests
# ---------------------------------------------------------------------------


class TestGeometryKindField:
    """DataEntry.geometry_kind parses correctly from GeometryKind enum values."""

    def test_default_geometry_kind_is_tabular(self, value_feature_default: ValueFeature) -> None:
        """Default geometry_kind is GeometryKind.TABULAR."""
        assert value_feature_default.geometry_kind == GeometryKind.TABULAR

    @pytest.mark.parametrize("kind", list(GeometryKind))
    def test_geometry_kind_set_from_enum(self, kind: GeometryKind) -> None:
        """geometry_kind accepts any GeometryKind enum value."""
        feat = ValueFeature(name="x", value=torch.zeros(4, 3), geometry_kind=kind)
        assert feat.geometry_kind == kind

    @pytest.mark.parametrize(
        "kind_str, expected",
        [
            ("tabular", GeometryKind.TABULAR),
            ("regular_grid", GeometryKind.REGULAR_GRID),
            ("sequence", GeometryKind.SEQUENCE),
            ("point_cloud", GeometryKind.POINT_CLOUD),
            ("graph", GeometryKind.GRAPH),
            ("mesh", GeometryKind.MESH),
        ],
    )
    def test_geometry_kind_parses_from_string(self, kind_str: str, expected: GeometryKind) -> None:
        """geometry_kind accepts the string value defined by GeometryKind (StrEnum)."""
        feat = ValueFeature(name="x", value=torch.zeros(4, 3), geometry_kind=kind_str)  # type: ignore[arg-type]
        assert feat.geometry_kind == expected

    def test_geometry_kind_excluded_from_model_dump(
        self, value_feature_default: ValueFeature
    ) -> None:
        """geometry_kind does not appear in model_dump() output (exclude=True)."""
        dumped = value_feature_default.model_dump()
        assert "geometry_kind" not in dumped
