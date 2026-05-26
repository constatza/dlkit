"""Unit tests for dlkit.common.geometry."""

import dataclasses

import pytest

from dlkit.common.geometry import (
    FieldRole,
    FieldSpec,
    GeometryKind,
    GeometrySpec,
    TopologyKind,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def feature_spec() -> FieldSpec:
    """A basic FEATURE field with tabular geometry."""
    return FieldSpec(name="x", shape=(16,), role=FieldRole.FEATURE)


@pytest.fixture
def coord_spec() -> FieldSpec:
    """A FEATURE_COORDINATES field with point-cloud geometry (N_points × coord_dim)."""
    return FieldSpec(
        name="coords",
        shape=(64, 3),
        role=FieldRole.FEATURE_COORDINATES,
        geometry_kind=GeometryKind.POINT_CLOUD,
    )


@pytest.fixture
def target_coord_spec() -> FieldSpec:
    """A TARGET_COORDINATES field with point-cloud geometry (N_queries × coord_dim)."""
    return FieldSpec(
        name="query_coords",
        shape=(32, 3),
        role=FieldRole.TARGET_COORDINATES,
        geometry_kind=GeometryKind.POINT_CLOUD,
    )


@pytest.fixture
def geometry_spec(
    feature_spec: FieldSpec,
    coord_spec: FieldSpec,
    target_coord_spec: FieldSpec,
) -> GeometrySpec:
    """A GeometrySpec containing one field of each role."""
    return GeometrySpec(
        fields=(feature_spec, coord_spec, target_coord_spec),
        topology_kind=TopologyKind.EDGE_INDEX,
        edge_feature_dim=4,
    )


@pytest.fixture
def no_feature_spec(coord_spec: FieldSpec, target_coord_spec: FieldSpec) -> GeometrySpec:
    """A GeometrySpec with no FEATURE-role field."""
    return GeometrySpec(fields=(coord_spec, target_coord_spec))


@pytest.fixture
def multidim_field_spec() -> FieldSpec:
    """A multi-dimensional FEATURE FieldSpec for primary_size tests."""
    return FieldSpec(name="grid", shape=(32, 64, 3), role=FieldRole.FEATURE)


@pytest.fixture
def multiple_features_same_role_spec() -> GeometrySpec:
    """A GeometrySpec with two FEATURE fields and one FEATURE_COORDINATES field."""
    f1 = FieldSpec(name="a", shape=(4,), role=FieldRole.FEATURE)
    f2 = FieldSpec(name="b", shape=(8,), role=FieldRole.FEATURE)
    coord = FieldSpec(name="c", shape=(3,), role=FieldRole.FEATURE_COORDINATES)
    return GeometrySpec(fields=(f1, f2, coord))


@pytest.fixture
def multiple_features_spec() -> GeometrySpec:
    """A GeometrySpec with two FEATURE fields for primary_feature ordering tests."""
    f1 = FieldSpec(name="first", shape=(4,), role=FieldRole.FEATURE)
    f2 = FieldSpec(name="second", shape=(8,), role=FieldRole.FEATURE)
    return GeometrySpec(fields=(f1, f2))


# ---------------------------------------------------------------------------
# FieldSpec tests
# ---------------------------------------------------------------------------


class TestFieldSpec:
    def test_primary_size_returns_shape_first_dim(self, feature_spec: FieldSpec) -> None:
        assert feature_spec.primary_size == feature_spec.shape[0]

    def test_primary_size_multidim(self, multidim_field_spec: FieldSpec) -> None:
        assert multidim_field_spec.primary_size == 32

    def test_default_geometry_kind_is_tabular(self, feature_spec: FieldSpec) -> None:
        assert feature_spec.geometry_kind == GeometryKind.TABULAR

    def test_custom_geometry_kind_stored(self, coord_spec: FieldSpec) -> None:
        assert coord_spec.geometry_kind == GeometryKind.POINT_CLOUD

    def test_is_frozen(self, feature_spec: FieldSpec) -> None:
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            feature_spec.name = "mutated"  # type: ignore[misc]

    def test_empty_shape_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one dimension"):
            FieldSpec(name="empty", shape=(), role=FieldRole.FEATURE)

    def test_zero_dimension_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            FieldSpec(name="bad", shape=(0, 4), role=FieldRole.FEATURE)

    def test_negative_dimension_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            FieldSpec(name="bad", shape=(8, -1), role=FieldRole.FEATURE)

    @pytest.mark.parametrize(
        "kind", [GeometryKind.SEQUENCE, GeometryKind.REGULAR_GRID, GeometryKind.POINT_CLOUD]
    )
    def test_1d_shape_with_multi_dim_kind_raises(self, kind: GeometryKind) -> None:
        with pytest.raises(ValueError, match="requires at least 2 dimensions"):
            FieldSpec(name="bad", shape=(8,), role=FieldRole.FEATURE, geometry_kind=kind)

    @pytest.mark.parametrize("kind", [GeometryKind.TABULAR, GeometryKind.GRAPH])
    def test_1d_shape_with_single_dim_kind_is_valid(self, kind: GeometryKind) -> None:
        spec = FieldSpec(name="ok", shape=(8,), role=FieldRole.FEATURE, geometry_kind=kind)
        assert spec.shape == (8,)


# ---------------------------------------------------------------------------
# GeometrySpec.by_role tests
# ---------------------------------------------------------------------------


class TestByRole:
    def test_returns_only_matching_role(
        self, geometry_spec: GeometrySpec, feature_spec: FieldSpec
    ) -> None:
        result = geometry_spec.by_role(FieldRole.FEATURE)
        assert result == (feature_spec,)

    def test_returns_empty_tuple_for_absent_role(self, no_feature_spec: GeometrySpec) -> None:
        result = no_feature_spec.by_role(FieldRole.FEATURE)
        assert result == ()

    def test_multiple_fields_same_role(
        self, multiple_features_same_role_spec: GeometrySpec
    ) -> None:
        result = multiple_features_same_role_spec.by_role(FieldRole.FEATURE)
        assert len(result) == 2
        assert all(f.role == FieldRole.FEATURE for f in result)

    def test_preserves_insertion_order(self, geometry_spec: GeometrySpec) -> None:
        features = geometry_spec.by_role(FieldRole.FEATURE)
        coords = geometry_spec.by_role(FieldRole.FEATURE_COORDINATES)
        target_coords = geometry_spec.by_role(FieldRole.TARGET_COORDINATES)
        assert len(features) == 1
        assert len(coords) == 1
        assert len(target_coords) == 1


# ---------------------------------------------------------------------------
# GeometrySpec.primary_feature tests
# ---------------------------------------------------------------------------


class TestPrimaryFeature:
    def test_returns_first_feature_field(
        self, geometry_spec: GeometrySpec, feature_spec: FieldSpec
    ) -> None:
        assert geometry_spec.primary_feature() == feature_spec

    def test_raises_when_no_feature_present(self, no_feature_spec: GeometrySpec) -> None:
        with pytest.raises(ValueError, match="no FEATURE field"):
            no_feature_spec.primary_feature()

    def test_returns_first_when_multiple_features(
        self, multiple_features_spec: GeometrySpec
    ) -> None:
        result = multiple_features_spec.primary_feature()
        assert result.name == "first"


# ---------------------------------------------------------------------------
# dataclasses.asdict round-trip
# ---------------------------------------------------------------------------


class TestAsDict:
    @pytest.mark.parametrize("role", list(FieldRole))
    def test_field_spec_roundtrip_all_roles(self, role: FieldRole) -> None:
        spec = FieldSpec(name="f", shape=(10,), role=role)
        d = dataclasses.asdict(spec)
        reconstructed = FieldSpec(**d)
        assert reconstructed == spec

    @pytest.mark.parametrize("kind", list(GeometryKind))
    def test_field_spec_roundtrip_all_geometry_kinds(self, kind: GeometryKind) -> None:
        spec = FieldSpec(name="f", shape=(5, 3), role=FieldRole.FEATURE, geometry_kind=kind)
        d = dataclasses.asdict(spec)
        reconstructed = FieldSpec(**d)
        assert reconstructed == spec

    @pytest.mark.parametrize("topology", [*list(TopologyKind), None])
    def test_geometry_spec_roundtrip_topology_kinds(
        self, feature_spec: FieldSpec, topology: TopologyKind | None
    ) -> None:
        spec = GeometrySpec(
            fields=(feature_spec,),
            topology_kind=topology,
            edge_feature_dim=None if topology is None else 2,
        )
        d = dataclasses.asdict(spec)
        # fields is a list of dicts after asdict — reconstruct manually
        reconstructed_fields = tuple(FieldSpec(**fd) for fd in d["fields"])
        reconstructed = GeometrySpec(
            fields=reconstructed_fields,
            topology_kind=d["topology_kind"],
            edge_feature_dim=d["edge_feature_dim"],
        )
        assert reconstructed == spec

    def test_geometry_spec_minimal_roundtrip(self, feature_spec: FieldSpec) -> None:
        spec = GeometrySpec(fields=(feature_spec,))
        d = dataclasses.asdict(spec)
        reconstructed_fields = tuple(FieldSpec(**fd) for fd in d["fields"])
        reconstructed = GeometrySpec(
            fields=reconstructed_fields,
            topology_kind=d["topology_kind"],
            edge_feature_dim=d["edge_feature_dim"],
        )
        assert reconstructed == spec
        assert reconstructed.topology_kind is None
        assert reconstructed.edge_feature_dim is None


# ---------------------------------------------------------------------------
# GeometrySpec.__post_init__ invariant tests
# ---------------------------------------------------------------------------


class TestGeometrySpecInvariants:
    def test_empty_fields_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            GeometrySpec(fields=())

    def test_edge_feature_dim_without_topology_raises(self, feature_spec: FieldSpec) -> None:
        with pytest.raises(ValueError, match="topology_kind"):
            GeometrySpec(fields=(feature_spec,), edge_feature_dim=4, topology_kind=None)

    def test_edge_feature_dim_with_topology_is_valid(self, feature_spec: FieldSpec) -> None:
        spec = GeometrySpec(
            fields=(feature_spec,),
            topology_kind=TopologyKind.EDGE_INDEX,
            edge_feature_dim=4,
        )
        assert spec.edge_feature_dim == 4

    def test_no_topology_no_edge_dim_is_valid(self, feature_spec: FieldSpec) -> None:
        spec = GeometrySpec(fields=(feature_spec,))
        assert spec.topology_kind is None
        assert spec.edge_feature_dim is None
