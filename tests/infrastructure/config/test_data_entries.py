"""Tests for DataEntry fields: model_input, field_role, and geometry_kind.

Covers:
- model_input bool field: True/False/non-bool rejection
- field_role: parses from string config values, defaults to FEATURE
- geometry_kind: parses from string config values, defaults to TABULAR
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
from pydantic import ValidationError

from dlkit.common.geometry import FieldRole, GeometryKind
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.entry_types import NpyEntry, ValueEntry

class TestModelInputField:
    """Tests for the model_input bool field on DataEntry subclasses."""

    def test_model_input_true_accepted(self) -> None:
        """True is accepted (include as model input)."""
        feat = ValueEntry(
            name="x", value=torch.zeros(4, 3), data_role=DataRole.FEATURE, model_input=True
        )
        assert feat.model_input is True

    def test_model_input_false_accepted(self) -> None:
        """False is accepted (exclude from model call)."""
        feat = ValueEntry(
            name="x", value=torch.zeros(4, 3), data_role=DataRole.FEATURE, model_input=False
        )
        assert feat.model_input is False

    def test_model_input_default_is_true(self) -> None:
        """Default model_input is True (include as model input)."""
        feat = ValueEntry(name="x", value=torch.zeros(4, 3), data_role=DataRole.FEATURE)
        assert feat.model_input is True

    def test_model_input_str_raises(self) -> None:
        """String model_input raises ValidationError (hard-cut to bool)."""
        with pytest.raises(ValidationError):
            ValueEntry(
                name="x",
                value=torch.zeros(4, 3),
                data_role=DataRole.FEATURE,
                model_input=cast("bool", "hidden"),
            )

    def test_model_input_int_raises(self) -> None:
        """Integer model_input raises ValidationError (hard-cut to bool)."""
        with pytest.raises(ValidationError):
            ValueEntry(
                name="x",
                value=torch.zeros(4, 3),
                data_role=DataRole.FEATURE,
                model_input=cast("bool", 0),
            )

    def test_model_input_none_raises(self) -> None:
        """None model_input raises ValidationError (hard-cut to bool)."""
        with pytest.raises(ValidationError):
            ValueEntry(
                name="x",
                value=torch.zeros(4, 3),
                data_role=DataRole.FEATURE,
                model_input=cast("bool", None),
            )

    def test_model_input_int_one_raises(self) -> None:
        """Integer 1 raises ValidationError (strict bool — no coercion from int)."""
        with pytest.raises(ValidationError):
            ValueEntry(
                name="x",
                value=torch.zeros(4, 3),
                data_role=DataRole.FEATURE,
                model_input=cast("bool", 1),
            )

    def test_model_input_kwarg_str_raises(self) -> None:
        """String 'kwarg' raises ValidationError (strict bool — no strings accepted)."""
        with pytest.raises(ValidationError):
            ValueEntry(
                name="x",
                value=torch.zeros(4, 3),
                data_role=DataRole.FEATURE,
                model_input=cast("bool", "kwarg"),
            )


# ---------------------------------------------------------------------------
# Fixtures for field_role / geometry_kind tests
# ---------------------------------------------------------------------------


@pytest.fixture
def value_entry_default() -> ValueEntry:
    """ValueEntry constructed with no explicit field_role or geometry_kind."""
    return ValueEntry(name="x", value=torch.zeros(4, 3), data_role=DataRole.FEATURE)


@pytest.fixture
def npy_entry_placeholder() -> NpyEntry:
    """NpyEntry placeholder with no path (no data source)."""
    return NpyEntry(name="x", data_role=DataRole.FEATURE)


# ---------------------------------------------------------------------------
# field_role tests
# ---------------------------------------------------------------------------


class TestFieldRoleField:
    """DataEntry.field_role parses correctly from FieldRole enum values."""

    def test_default_field_role_is_feature(self, value_entry_default: ValueEntry) -> None:
        """Default field_role is FieldRole.FEATURE."""
        assert value_entry_default.field_role == FieldRole.FEATURE

    @pytest.mark.parametrize("role", list(FieldRole))
    def test_field_role_set_from_enum(self, role: FieldRole) -> None:
        """field_role accepts any FieldRole enum value."""
        feat = ValueEntry(
            name="x", value=torch.zeros(4, 3), data_role=DataRole.FEATURE, field_role=role
        )
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
        feat = ValueEntry(
            name="x",
            value=torch.zeros(4, 3),
            data_role=DataRole.FEATURE,
            field_role=cast("FieldRole", role_str),
        )
        assert feat.field_role == expected

    def test_field_role_excluded_from_model_dump(self, value_entry_default: ValueEntry) -> None:
        """field_role does not appear in model_dump() output (exclude=True)."""
        dumped = value_entry_default.model_dump()
        assert "field_role" not in dumped


# ---------------------------------------------------------------------------
# geometry_kind tests
# ---------------------------------------------------------------------------


class TestGeometryKindField:
    """DataEntry.geometry_kind parses correctly from GeometryKind enum values."""

    def test_default_geometry_kind_is_tabular(self, value_entry_default: ValueEntry) -> None:
        """Default geometry_kind is GeometryKind.TABULAR."""
        assert value_entry_default.geometry_kind == GeometryKind.TABULAR

    @pytest.mark.parametrize("kind", list(GeometryKind))
    def test_geometry_kind_set_from_enum(self, kind: GeometryKind) -> None:
        """geometry_kind accepts any GeometryKind enum value."""
        feat = ValueEntry(
            name="x", value=torch.zeros(4, 3), data_role=DataRole.FEATURE, geometry_kind=kind
        )
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
        feat = ValueEntry(
            name="x",
            value=torch.zeros(4, 3),
            data_role=DataRole.FEATURE,
            geometry_kind=cast("GeometryKind", kind_str),
        )
        assert feat.geometry_kind == expected

    def test_geometry_kind_excluded_from_model_dump(self, value_entry_default: ValueEntry) -> None:
        """geometry_kind does not appear in model_dump() output (exclude=True)."""
        dumped = value_entry_default.model_dump()
        assert "geometry_kind" not in dumped
