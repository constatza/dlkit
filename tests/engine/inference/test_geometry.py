"""Unit tests for dlkit.engine.inference.geometry.

Covers the 3-case fallback in infer_geometry_from_checkpoint and
the _geometry_from_dict reconstruction helper.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import MagicMock

import pytest

from dlkit.common.geometry import FieldRole, FieldSpec, GeometryKind, GeometrySpec, TopologyKind
from dlkit.engine.inference.geometry import _geometry_from_dict, infer_geometry_from_checkpoint

# ---------------------------------------------------------------------------
# Fixtures — checkpoint shapes
# ---------------------------------------------------------------------------


@pytest.fixture
def single_feature_spec() -> FieldSpec:
    """A single FEATURE FieldSpec with tabular geometry and shape (16,)."""
    return FieldSpec(name="x", shape=(16,), role=FieldRole.FEATURE)


@pytest.fixture
def multi_field_spec() -> tuple[FieldSpec, ...]:
    """Two FieldSpecs: one FEATURE and one TARGET_COORDINATES."""
    return (
        FieldSpec(name="sensor", shape=(100,), role=FieldRole.FEATURE),
        FieldSpec(
            name="query",
            shape=(2,),
            role=FieldRole.TARGET_COORDINATES,
            geometry_kind=GeometryKind.REGULAR_GRID,
        ),
    )


@pytest.fixture
def geometry_spec_simple(single_feature_spec: FieldSpec) -> GeometrySpec:
    """Minimal GeometrySpec with one FEATURE field."""
    return GeometrySpec(fields=(single_feature_spec,))


@pytest.fixture
def geometry_spec_with_topology(multi_field_spec: tuple[FieldSpec, ...]) -> GeometrySpec:
    """GeometrySpec with two fields and EDGE_INDEX topology."""
    return GeometrySpec(
        fields=multi_field_spec,
        topology_kind=TopologyKind.EDGE_INDEX,
        edge_feature_dim=4,
    )


@pytest.fixture
def geometry_dict_simple(geometry_spec_simple: GeometrySpec) -> dict[str, Any]:
    """dataclasses.asdict() of geometry_spec_simple."""
    return dataclasses.asdict(geometry_spec_simple)


@pytest.fixture
def geometry_dict_with_topology(geometry_spec_with_topology: GeometrySpec) -> dict[str, Any]:
    """dataclasses.asdict() of geometry_spec_with_topology."""
    return dataclasses.asdict(geometry_spec_with_topology)


@pytest.fixture
def checkpoint_with_geometry(geometry_spec_simple: GeometrySpec) -> dict[str, Any]:
    """Checkpoint dict carrying dlkit_metadata.geometry (Case 1)."""
    return {
        "dlkit_metadata": {
            "geometry": dataclasses.asdict(geometry_spec_simple),
        }
    }


@pytest.fixture
def checkpoint_without_dlkit_metadata() -> dict[str, Any]:
    """Checkpoint dict with no dlkit_metadata key (Case 2 — external checkpoint)."""
    return {"state_dict": {}, "epoch": 5}


@pytest.fixture
def checkpoint_with_metadata_no_geometry() -> dict[str, Any]:
    """Checkpoint dict with dlkit_metadata but no geometry (Case 3)."""
    return {
        "dlkit_metadata": {
            "model_name": "SomeModel",
        }
    }


@pytest.fixture
def legacy_checkpoint_with_shape_summary() -> dict[str, Any]:
    """Checkpoint dict with dlkit_metadata.shape_summary (legacy format)."""
    return {
        "dlkit_metadata": {
            "shape_summary": {
                "in_shapes": [[8], [4, 16]],
                "out_shapes": [[3]],
            }
        }
    }


# ---------------------------------------------------------------------------
# Tests: _geometry_from_dict reconstruction
# ---------------------------------------------------------------------------


class TestGeometryFromDict:
    """_geometry_from_dict correctly reconstructs all field types."""

    def test_single_tabular_field(
        self, geometry_dict_simple: dict[str, Any], geometry_spec_simple: GeometrySpec
    ) -> None:
        """Reconstructs a simple single-field GeometrySpec without topology."""
        result = _geometry_from_dict(geometry_dict_simple)

        assert result == geometry_spec_simple

    def test_multiple_fields_with_topology(
        self,
        geometry_dict_with_topology: dict[str, Any],
        geometry_spec_with_topology: GeometrySpec,
    ) -> None:
        """Reconstructs GeometrySpec with two fields and EDGE_INDEX topology."""
        result = _geometry_from_dict(geometry_dict_with_topology)

        assert result == geometry_spec_with_topology

    def test_field_roles_restored(self, geometry_dict_with_topology: dict[str, Any]) -> None:
        """FieldRole enum values are correctly restored from string."""
        result = _geometry_from_dict(geometry_dict_with_topology)

        assert result.fields[0].role == FieldRole.FEATURE
        assert result.fields[1].role == FieldRole.TARGET_COORDINATES

    def test_geometry_kinds_restored(self, geometry_dict_with_topology: dict[str, Any]) -> None:
        """GeometryKind enum values are correctly restored from string."""
        result = _geometry_from_dict(geometry_dict_with_topology)

        assert result.fields[0].geometry_kind == GeometryKind.TABULAR
        assert result.fields[1].geometry_kind == GeometryKind.REGULAR_GRID

    def test_topology_kind_none_when_absent(self, geometry_dict_simple: dict[str, Any]) -> None:
        """topology_kind is None when not present in the dict."""
        result = _geometry_from_dict(geometry_dict_simple)

        assert result.topology_kind is None

    def test_topology_kind_edge_index_restored(
        self, geometry_dict_with_topology: dict[str, Any]
    ) -> None:
        """TopologyKind.EDGE_INDEX is correctly restored from string."""
        result = _geometry_from_dict(geometry_dict_with_topology)

        assert result.topology_kind == TopologyKind.EDGE_INDEX

    def test_edge_feature_dim_restored(self, geometry_dict_with_topology: dict[str, Any]) -> None:
        """edge_feature_dim is correctly restored."""
        result = _geometry_from_dict(geometry_dict_with_topology)

        assert result.edge_feature_dim == 4

    def test_shapes_are_tuples(self, geometry_dict_simple: dict[str, Any]) -> None:
        """Field shapes are restored as tuples, not lists."""
        result = _geometry_from_dict(geometry_dict_simple)

        assert isinstance(result.fields[0].shape, tuple)

    @pytest.mark.parametrize("role", list(FieldRole))
    def test_all_field_roles_round_trip(self, role: FieldRole) -> None:
        """Every FieldRole survives a dict round-trip."""
        original = GeometrySpec(fields=(FieldSpec(name="f", shape=(5,), role=role),))
        d = dataclasses.asdict(original)
        restored = _geometry_from_dict(d)

        assert restored.fields[0].role == role

    @pytest.mark.parametrize("kind", list(GeometryKind))
    def test_all_geometry_kinds_round_trip(self, kind: GeometryKind) -> None:
        """Every GeometryKind survives a dict round-trip."""
        original = GeometrySpec(
            fields=(FieldSpec(name="f", shape=(5,), role=FieldRole.FEATURE, geometry_kind=kind),)
        )
        d = dataclasses.asdict(original)
        restored = _geometry_from_dict(d)

        assert restored.fields[0].geometry_kind == kind


# ---------------------------------------------------------------------------
# Tests: infer_geometry_from_checkpoint — Case 1 (geometry in metadata)
# ---------------------------------------------------------------------------


class TestInferGeometryCase1:
    """Case 1: dlkit_metadata.geometry present → GeometrySpec restored."""

    def test_returns_geometry_spec(
        self,
        checkpoint_with_geometry: dict[str, Any],
        geometry_spec_simple: GeometrySpec,
    ) -> None:
        """Returns a GeometrySpec when geometry is serialised in the checkpoint."""
        result = infer_geometry_from_checkpoint(checkpoint_with_geometry)

        assert result == geometry_spec_simple

    def test_returns_correct_field_count(self, checkpoint_with_geometry: dict[str, Any]) -> None:
        """Restored GeometrySpec has the expected number of fields."""
        result = infer_geometry_from_checkpoint(checkpoint_with_geometry)

        assert result is not None
        assert len(result.fields) == 1

    def test_restored_field_has_correct_shape(
        self, checkpoint_with_geometry: dict[str, Any]
    ) -> None:
        """Restored primary feature has the original shape."""
        result = infer_geometry_from_checkpoint(checkpoint_with_geometry)

        assert result is not None
        assert result.fields[0].shape == (16,)

    def test_dataset_not_needed_for_case1(self, checkpoint_with_geometry: dict[str, Any]) -> None:
        """Dataset is not consulted when geometry is already in the checkpoint."""
        mock_dataset = MagicMock()

        result = infer_geometry_from_checkpoint(checkpoint_with_geometry, dataset=mock_dataset)

        assert result is not None
        mock_dataset.__getitem__.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: infer_geometry_from_checkpoint — Case 2 (no dlkit_metadata)
# ---------------------------------------------------------------------------


class TestInferGeometryCase2:
    """Case 2: dlkit_metadata absent → returns None (external checkpoint)."""

    def test_returns_none_for_external_checkpoint(
        self, checkpoint_without_dlkit_metadata: dict[str, Any]
    ) -> None:
        """Returns None when dlkit_metadata is entirely absent."""
        result = infer_geometry_from_checkpoint(checkpoint_without_dlkit_metadata)

        assert result is None

    def test_returns_none_for_empty_checkpoint(self) -> None:
        """Returns None for a completely empty checkpoint dict."""
        result = infer_geometry_from_checkpoint({})

        assert result is None

    def test_dataset_ignored_when_no_metadata(
        self, checkpoint_without_dlkit_metadata: dict[str, Any]
    ) -> None:
        """Dataset is not consulted when there is no dlkit_metadata at all."""
        mock_dataset = MagicMock()

        result = infer_geometry_from_checkpoint(
            checkpoint_without_dlkit_metadata, dataset=mock_dataset
        )

        assert result is None
        mock_dataset.__getitem__.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: infer_geometry_from_checkpoint — Case 3 (metadata, no geometry)
# ---------------------------------------------------------------------------


class TestInferGeometryCase3:
    """Case 3: dlkit_metadata present but no geometry → None when no dataset."""

    def test_returns_none_when_no_dataset(
        self, checkpoint_with_metadata_no_geometry: dict[str, Any]
    ) -> None:
        """Returns None when metadata is present but geometry is absent and no dataset given."""
        result = infer_geometry_from_checkpoint(checkpoint_with_metadata_no_geometry)

        assert result is None

    def test_returns_none_when_dataset_is_none(
        self, checkpoint_with_metadata_no_geometry: dict[str, Any]
    ) -> None:
        """Returns None when metadata is present but dataset=None is explicitly passed."""
        result = infer_geometry_from_checkpoint(checkpoint_with_metadata_no_geometry, dataset=None)

        assert result is None


# ---------------------------------------------------------------------------
# Tests: legacy shape_summary fallback
# ---------------------------------------------------------------------------


class TestLegacyShapeSummaryFallback:
    """dlkit_metadata with shape_summary (old format) synthesises a GeometrySpec."""

    def test_returns_geometry_from_shape_summary(
        self, legacy_checkpoint_with_shape_summary: dict[str, Any]
    ) -> None:
        """GeometrySpec is built from in_shapes in the legacy shape_summary."""
        result = infer_geometry_from_checkpoint(legacy_checkpoint_with_shape_summary)

        assert result is not None

    def test_field_count_matches_in_shapes(
        self, legacy_checkpoint_with_shape_summary: dict[str, Any]
    ) -> None:
        """Number of FieldSpecs equals the number of entries in in_shapes."""
        result = infer_geometry_from_checkpoint(legacy_checkpoint_with_shape_summary)

        assert result is not None
        assert len(result.fields) == 2

    def test_tabular_shape_yields_tabular_kind(
        self, legacy_checkpoint_with_shape_summary: dict[str, Any]
    ) -> None:
        """A 1-D in_shape (rank 1) produces GeometryKind.TABULAR."""
        result = infer_geometry_from_checkpoint(legacy_checkpoint_with_shape_summary)

        assert result is not None
        assert result.fields[0].geometry_kind == GeometryKind.TABULAR
        assert result.fields[0].shape == (8,)

    def test_multidim_shape_yields_regular_grid_kind(
        self, legacy_checkpoint_with_shape_summary: dict[str, Any]
    ) -> None:
        """A multi-dimensional in_shape (rank > 1) produces GeometryKind.REGULAR_GRID."""
        result = infer_geometry_from_checkpoint(legacy_checkpoint_with_shape_summary)

        assert result is not None
        assert result.fields[1].geometry_kind == GeometryKind.REGULAR_GRID
        assert result.fields[1].shape == (4, 16)

    def test_all_fields_have_feature_role(
        self, legacy_checkpoint_with_shape_summary: dict[str, Any]
    ) -> None:
        """All synthesised FieldSpecs have FieldRole.FEATURE."""
        result = infer_geometry_from_checkpoint(legacy_checkpoint_with_shape_summary)

        assert result is not None
        assert all(f.role == FieldRole.FEATURE for f in result.fields)

    def test_no_topology_kind_in_legacy_geometry(
        self, legacy_checkpoint_with_shape_summary: dict[str, Any]
    ) -> None:
        """Legacy shape_summary does not set a topology_kind."""
        result = infer_geometry_from_checkpoint(legacy_checkpoint_with_shape_summary)

        assert result is not None
        assert result.topology_kind is None
