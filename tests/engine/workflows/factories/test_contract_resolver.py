"""Tests for dlkit.engine.workflows.factories.contract_resolver.resolve_contract."""

from __future__ import annotations

import pytest

from dlkit.common.geometry import FieldRole, FieldSpec, GeometryKind, GeometrySpec, TopologyKind
from dlkit.domain.nn.contracts import (
    BranchTrunkSpec,
    GraphContractSpec,
    GridOperatorSpec,
    SequenceSpec,
    TabulaRSpec,
)
from dlkit.engine.workflows.factories.contract_resolver import (
    ContractInferenceError,
    resolve_contract,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tabular_geometry() -> GeometrySpec:
    """GeometrySpec with a single TABULAR FEATURE field of shape (8,).

    Returns:
        GeometrySpec for a flat tabular feature.
    """
    return GeometrySpec(
        fields=(
            FieldSpec(
                name="x", shape=(8,), role=FieldRole.FEATURE, geometry_kind=GeometryKind.TABULAR
            ),
        )
    )


@pytest.fixture
def sequence_geometry() -> GeometrySpec:
    """GeometrySpec with a single SEQUENCE FEATURE field of shape (16, 32).

    Returns:
        GeometrySpec for a sequence feature (channels=16, seq_len=32).
    """
    return GeometrySpec(
        fields=(
            FieldSpec(
                name="seq",
                shape=(16, 32),
                role=FieldRole.FEATURE,
                geometry_kind=GeometryKind.SEQUENCE,
            ),
        )
    )


@pytest.fixture
def grid_geometry() -> GeometrySpec:
    """GeometrySpec with a REGULAR_GRID FEATURE field of shape (4, 64, 64).

    Returns:
        GeometrySpec for a 2-D spatial grid (channels=4, height=64, width=64).
    """
    return GeometrySpec(
        fields=(
            FieldSpec(
                name="grid",
                shape=(4, 64, 64),
                role=FieldRole.FEATURE,
                geometry_kind=GeometryKind.REGULAR_GRID,
            ),
        )
    )


@pytest.fixture
def tabular_with_coords_geometry() -> GeometrySpec:
    """GeometrySpec with FEATURE (tabular) and TARGET_COORDINATES fields.

    Returns:
        GeometrySpec suitable for a DeepONet branch-trunk setup.
    """
    return GeometrySpec(
        fields=(
            FieldSpec(
                name="sensor",
                shape=(100,),
                role=FieldRole.FEATURE,
                geometry_kind=GeometryKind.TABULAR,
            ),
            FieldSpec(
                name="coords",
                shape=(2,),
                role=FieldRole.TARGET_COORDINATES,
                geometry_kind=GeometryKind.TABULAR,
            ),
        )
    )


@pytest.fixture
def graph_geometry() -> GeometrySpec:
    """GeometrySpec with a single GRAPH FEATURE field of shape (32,).

    Returns:
        GeometrySpec for a GNN with node features of width 32.
    """
    return GeometrySpec(
        fields=(
            FieldSpec(
                name="node_x", shape=(32,), role=FieldRole.FEATURE, geometry_kind=GeometryKind.GRAPH
            ),
        ),
        topology_kind=TopologyKind.EDGE_INDEX,
        edge_feature_dim=None,
    )


@pytest.fixture
def graph_geometry_with_edges() -> GeometrySpec:
    """GeometrySpec for a GNN with 8-dimensional edge features.

    Returns:
        GeometrySpec with edge_feature_dim=8.
    """
    return GeometrySpec(
        fields=(
            FieldSpec(
                name="node_x", shape=(32,), role=FieldRole.FEATURE, geometry_kind=GeometryKind.GRAPH
            ),
        ),
        topology_kind=TopologyKind.EDGE_INDEX,
        edge_feature_dim=8,
    )


@pytest.fixture
def point_cloud_geometry() -> GeometrySpec:
    """GeometrySpec with a POINT_CLOUD FEATURE field of shape (3, 1024).

    Returns:
        GeometrySpec for a point cloud (channels=3, points=1024).
    """
    return GeometrySpec(
        fields=(
            FieldSpec(
                name="x",
                shape=(3, 1024),
                role=FieldRole.FEATURE,
                geometry_kind=GeometryKind.POINT_CLOUD,
            ),
        )
    )


@pytest.fixture
def mesh_geometry() -> GeometrySpec:
    """GeometrySpec with a MESH FEATURE field.

    Returns:
        GeometrySpec for an unsupported mesh geometry.
    """
    return GeometrySpec(
        fields=(
            FieldSpec(
                name="mesh", shape=(10,), role=FieldRole.FEATURE, geometry_kind=GeometryKind.MESH
            ),
        )
    )


@pytest.fixture
def grid_with_coords_geometry() -> GeometrySpec:
    """GeometrySpec with REGULAR_GRID FEATURE and TARGET_COORDINATES fields.

    Returns:
        GeometrySpec for a DeepONet branch-trunk setup with grid sensor input.
    """
    return GeometrySpec(
        fields=(
            FieldSpec(
                name="sensor",
                shape=(1, 100),
                role=FieldRole.FEATURE,
                geometry_kind=GeometryKind.REGULAR_GRID,
            ),
            FieldSpec(
                name="coords",
                shape=(1, 3),
                role=FieldRole.TARGET_COORDINATES,
                geometry_kind=GeometryKind.REGULAR_GRID,
            ),
        )
    )


@pytest.fixture
def point_cloud_with_coords_geometry() -> GeometrySpec:
    """GeometrySpec with POINT_CLOUD FEATURE and TARGET_COORDINATES fields.

    Returns:
        GeometrySpec for a DeepONet branch-trunk setup with point-cloud sensor input.
    """
    return GeometrySpec(
        fields=(
            FieldSpec(
                name="sensor",
                shape=(64, 3),
                role=FieldRole.FEATURE,
                geometry_kind=GeometryKind.POINT_CLOUD,
            ),
            FieldSpec(
                name="query",
                shape=(32, 2),
                role=FieldRole.TARGET_COORDINATES,
                geometry_kind=GeometryKind.POINT_CLOUD,
            ),
        )
    )


@pytest.fixture
def no_feature_geometry() -> GeometrySpec:
    """GeometrySpec with only a TARGET_COORDINATES field (no FEATURE field).

    Returns:
        GeometrySpec missing a FEATURE-role field.
    """
    return GeometrySpec(
        fields=(
            FieldSpec(
                name="coords",
                shape=(2,),
                role=FieldRole.TARGET_COORDINATES,
                geometry_kind=GeometryKind.TABULAR,
            ),
        )
    )


# ---------------------------------------------------------------------------
# Tests: correct contract variant returned
# ---------------------------------------------------------------------------


class TestTabularDispatch:
    """resolve_contract dispatches TABULAR → TabulaRSpec."""

    def test_returns_tabular_spec(self, tabular_geometry: GeometrySpec) -> None:
        """TABULAR geometry without coords produces TabulaRSpec.

        Args:
            tabular_geometry: Flat tabular feature fixture.
        """
        result = resolve_contract(tabular_geometry, output_shapes=((3,),))

        assert isinstance(result, TabulaRSpec)

    def test_in_shape_matches_feature_shape(self, tabular_geometry: GeometrySpec) -> None:
        """TabulaRSpec.in_shape equals the primary feature shape.

        Args:
            tabular_geometry: Flat tabular feature fixture.
        """
        result = resolve_contract(tabular_geometry, output_shapes=((3,),))

        assert result.in_shape == (8,)  # type: ignore[union-attr]

    def test_out_shape_from_output_shapes(self, tabular_geometry: GeometrySpec) -> None:
        """TabulaRSpec.out_shape equals output_shapes[0].

        Args:
            tabular_geometry: Flat tabular feature fixture.
        """
        result = resolve_contract(tabular_geometry, output_shapes=((5,),))

        assert result.out_shape == (5,)  # type: ignore[union-attr]

    def test_raises_when_no_output_shapes(self, tabular_geometry: GeometrySpec) -> None:
        """resolve_contract raises WorkflowError for tabular geometry with no output shapes.

        Legacy checkpoints that lack contract metadata cannot recover the output
        shape from geometry alone. The user must retrain to generate a compatible
        checkpoint.

        Args:
            tabular_geometry: Flat tabular feature fixture.
        """
        from dlkit.common.errors import WorkflowError

        with pytest.raises(WorkflowError, match="Cannot resolve tabular output shape"):
            resolve_contract(tabular_geometry)


class TestSequenceDispatch:
    """resolve_contract dispatches SEQUENCE → SequenceSpec."""

    def test_returns_sequence_spec(self, sequence_geometry: GeometrySpec) -> None:
        """SEQUENCE geometry produces SequenceSpec.

        Args:
            sequence_geometry: Sequence feature fixture.
        """
        result = resolve_contract(sequence_geometry, output_shapes=((4, 32),))

        assert isinstance(result, SequenceSpec)

    def test_in_channels_from_primary_size(self, sequence_geometry: GeometrySpec) -> None:
        """SequenceSpec.in_channels equals primary_feature().primary_size.

        Args:
            sequence_geometry: Sequence feature fixture.
        """
        result = resolve_contract(sequence_geometry, output_shapes=((4, 32),))

        assert result.in_channels == 16  # type: ignore[union-attr]

    def test_seq_len_from_shape(self, sequence_geometry: GeometrySpec) -> None:
        """SequenceSpec.seq_len equals shape[1] of the primary feature.

        Args:
            sequence_geometry: Sequence feature fixture.
        """
        result = resolve_contract(sequence_geometry, output_shapes=((4, 32),))

        assert result.seq_len == 32  # type: ignore[union-attr]

    def test_out_channels_from_output_shapes(self, sequence_geometry: GeometrySpec) -> None:
        """SequenceSpec.out_channels equals output_shapes[0][0].

        Args:
            sequence_geometry: Sequence feature fixture.
        """
        result = resolve_contract(sequence_geometry, output_shapes=((7, 32),))

        assert result.out_channels == 7  # type: ignore[union-attr]


class TestGridDispatch:
    """resolve_contract dispatches REGULAR_GRID → GridOperatorSpec."""

    def test_returns_grid_operator_spec(self, grid_geometry: GeometrySpec) -> None:
        """REGULAR_GRID geometry without coords produces GridOperatorSpec.

        Args:
            grid_geometry: 2-D grid feature fixture.
        """
        result = resolve_contract(grid_geometry, output_shapes=((2,),))

        assert isinstance(result, GridOperatorSpec)

    def test_in_channels_from_primary_size(self, grid_geometry: GeometrySpec) -> None:
        """GridOperatorSpec.in_channels equals primary_feature().primary_size.

        Args:
            grid_geometry: 2-D grid feature fixture.
        """
        result = resolve_contract(grid_geometry, output_shapes=((2,),))

        assert result.in_channels == 4  # type: ignore[union-attr]

    def test_out_channels_from_output_shapes(self, grid_geometry: GeometrySpec) -> None:
        """GridOperatorSpec.out_channels equals output_shapes[0][0].

        Args:
            grid_geometry: 2-D grid feature fixture.
        """
        result = resolve_contract(grid_geometry, output_shapes=((6,),))

        assert result.out_channels == 6  # type: ignore[union-attr]

    def test_spatial_shape_drops_channel_dim(self, grid_geometry: GeometrySpec) -> None:
        """GridOperatorSpec.spatial_shape equals shape[1:] of the primary feature.

        Args:
            grid_geometry: 2-D grid feature fixture.
        """
        result = resolve_contract(grid_geometry, output_shapes=((2,),))

        assert result.spatial_shape == (64, 64)  # type: ignore[union-attr]

    def test_point_cloud_returns_grid_operator_spec(
        self, point_cloud_geometry: GeometrySpec
    ) -> None:
        """POINT_CLOUD geometry without coords produces GridOperatorSpec.

        Args:
            point_cloud_geometry: Point cloud feature fixture (3, 1024).
        """
        result = resolve_contract(point_cloud_geometry, output_shapes=((2,),))

        assert isinstance(result, GridOperatorSpec)


class TestBranchTrunkDispatch:
    """resolve_contract dispatches TABULAR + TARGET_COORDINATES → BranchTrunkSpec."""

    def test_returns_branch_trunk_spec(self, tabular_with_coords_geometry: GeometrySpec) -> None:
        """TABULAR geometry with TARGET_COORDINATES produces BranchTrunkSpec.

        Args:
            tabular_with_coords_geometry: Branch-trunk fixture.
        """
        result = resolve_contract(tabular_with_coords_geometry, output_shapes=((1,),))

        assert isinstance(result, BranchTrunkSpec)

    def test_branch_shape_is_primary_feature_shape(
        self, tabular_with_coords_geometry: GeometrySpec
    ) -> None:
        """BranchTrunkSpec.branch_shape equals the primary FEATURE shape.

        Args:
            tabular_with_coords_geometry: Branch-trunk fixture.
        """
        result = resolve_contract(tabular_with_coords_geometry, output_shapes=((1,),))

        assert result.branch_shape == (100,)  # type: ignore[union-attr]

    def test_query_shape_is_coords_shape(self, tabular_with_coords_geometry: GeometrySpec) -> None:
        """BranchTrunkSpec.query_shape equals the TARGET_COORDINATES field shape.

        Args:
            tabular_with_coords_geometry: Branch-trunk fixture.
        """
        result = resolve_contract(tabular_with_coords_geometry, output_shapes=((1,),))

        assert result.query_shape == (2,)  # type: ignore[union-attr]

    def test_out_features_from_output_shapes(
        self, tabular_with_coords_geometry: GeometrySpec
    ) -> None:
        """Single-query tabular targets use their sole feature width.

        Args:
            tabular_with_coords_geometry: Branch-trunk fixture.
        """
        result = resolve_contract(tabular_with_coords_geometry, output_shapes=((5,),))

        assert result.out_features == 5  # type: ignore[union-attr]


class TestGraphDispatch:
    """resolve_contract dispatches GRAPH → GraphContractSpec."""

    def test_returns_graph_contract_spec(self, graph_geometry: GeometrySpec) -> None:
        """GRAPH geometry produces GraphContractSpec.

        Args:
            graph_geometry: GNN feature fixture without edge features.
        """
        result = resolve_contract(graph_geometry)

        assert isinstance(result, GraphContractSpec)

    def test_in_channels_from_primary_size(self, graph_geometry: GeometrySpec) -> None:
        """GraphContractSpec.in_channels equals primary_feature().primary_size.

        Args:
            graph_geometry: GNN feature fixture.
        """
        result = resolve_contract(graph_geometry)

        assert result.in_channels == 32  # type: ignore[union-attr]

    def test_out_channels_same_as_in_channels(self, graph_geometry: GeometrySpec) -> None:
        """GraphContractSpec.out_channels defaults to primary_size (same as in_channels).

        Args:
            graph_geometry: GNN feature fixture.
        """
        result = resolve_contract(graph_geometry)

        assert result.out_channels == result.in_channels  # type: ignore[union-attr]

    def test_edge_dim_none_when_no_edges(self, graph_geometry: GeometrySpec) -> None:
        """GraphContractSpec.edge_dim is None when geometry.edge_feature_dim is None.

        Args:
            graph_geometry: GNN feature fixture without edge features.
        """
        result = resolve_contract(graph_geometry)

        assert result.edge_dim is None  # type: ignore[union-attr]

    def test_edge_dim_propagated(self, graph_geometry_with_edges: GeometrySpec) -> None:
        """GraphContractSpec.edge_dim equals geometry.edge_feature_dim.

        Args:
            graph_geometry_with_edges: GNN fixture with 8-dim edge features.
        """
        result = resolve_contract(graph_geometry_with_edges)

        assert result.edge_dim == 8  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Tests: REGULAR_GRID / POINT_CLOUD + TARGET_COORDINATES → BranchTrunkSpec
# ---------------------------------------------------------------------------


class TestGridWithCoordsDispatch:
    """resolve_contract dispatches REGULAR_GRID + TARGET_COORDINATES → BranchTrunkSpec."""

    def test_returns_branch_trunk_spec(self, grid_with_coords_geometry: GeometrySpec) -> None:
        """REGULAR_GRID geometry with TARGET_COORDINATES produces BranchTrunkSpec.

        Args:
            grid_with_coords_geometry: Grid branch-trunk fixture.
        """
        result = resolve_contract(grid_with_coords_geometry, output_shapes=((1,),))

        assert isinstance(result, BranchTrunkSpec)

    def test_branch_shape_is_primary_feature_shape(
        self, grid_with_coords_geometry: GeometrySpec
    ) -> None:
        """BranchTrunkSpec.branch_shape equals the primary FEATURE shape.

        Args:
            grid_with_coords_geometry: Grid branch-trunk fixture.
        """
        result = resolve_contract(grid_with_coords_geometry, output_shapes=((1,),))

        assert result.branch_shape == (1, 100)  # type: ignore[union-attr]

    def test_query_shape_is_target_coord_shape(
        self, grid_with_coords_geometry: GeometrySpec
    ) -> None:
        """BranchTrunkSpec.query_shape equals the TARGET_COORDINATES field shape.

        Args:
            grid_with_coords_geometry: Grid branch-trunk fixture.
        """
        result = resolve_contract(grid_with_coords_geometry, output_shapes=((1,),))

        assert result.query_shape == (1, 3)  # type: ignore[union-attr]

    def test_out_features_from_output_shapes(self, grid_with_coords_geometry: GeometrySpec) -> None:
        """Query-shaped scalar targets resolve to one output feature.

        Args:
            grid_with_coords_geometry: Grid branch-trunk fixture.
        """
        result = resolve_contract(grid_with_coords_geometry, output_shapes=((8,),))

        assert result.out_features == 1  # type: ignore[union-attr]

    def test_vector_outputs_use_last_target_dimension(
        self, grid_with_coords_geometry: GeometrySpec
    ) -> None:
        """Query-shaped vector targets resolve to their feature dimension."""
        result = resolve_contract(grid_with_coords_geometry, output_shapes=((8, 3),))

        assert result.out_features == 3  # type: ignore[union-attr]


class TestPointCloudWithCoordsDispatch:
    """resolve_contract dispatches POINT_CLOUD + TARGET_COORDINATES → BranchTrunkSpec."""

    def test_returns_branch_trunk_spec(
        self, point_cloud_with_coords_geometry: GeometrySpec
    ) -> None:
        """POINT_CLOUD geometry with TARGET_COORDINATES produces BranchTrunkSpec.

        Args:
            point_cloud_with_coords_geometry: Point-cloud branch-trunk fixture.
        """
        result = resolve_contract(point_cloud_with_coords_geometry, output_shapes=((1,),))

        assert isinstance(result, BranchTrunkSpec)

    def test_branch_shape_is_primary_feature_shape(
        self, point_cloud_with_coords_geometry: GeometrySpec
    ) -> None:
        """BranchTrunkSpec.branch_shape equals the primary FEATURE shape.

        Args:
            point_cloud_with_coords_geometry: Point-cloud branch-trunk fixture.
        """
        result = resolve_contract(point_cloud_with_coords_geometry, output_shapes=((1,),))

        assert result.branch_shape == (64, 3)  # type: ignore[union-attr]

    def test_query_shape_is_target_coord_shape(
        self, point_cloud_with_coords_geometry: GeometrySpec
    ) -> None:
        """BranchTrunkSpec.query_shape equals the TARGET_COORDINATES field shape.

        Args:
            point_cloud_with_coords_geometry: Point-cloud branch-trunk fixture.
        """
        result = resolve_contract(point_cloud_with_coords_geometry, output_shapes=((1,),))

        assert result.query_shape == (32, 2)  # type: ignore[union-attr]

    def test_out_features_default_when_no_output_shapes(
        self, point_cloud_with_coords_geometry: GeometrySpec
    ) -> None:
        """BranchTrunkSpec.out_features defaults to 1 when output_shapes is empty.

        Args:
            point_cloud_with_coords_geometry: Point-cloud branch-trunk fixture.
        """
        result = resolve_contract(point_cloud_with_coords_geometry)

        assert result.out_features == 1  # type: ignore[union-attr]

    def test_point_cloud_vector_outputs_use_last_target_dimension(
        self, point_cloud_with_coords_geometry: GeometrySpec
    ) -> None:
        """Point-cloud query targets use the last target dimension as output width."""
        result = resolve_contract(point_cloud_with_coords_geometry, output_shapes=((32, 4),))

        assert result.out_features == 4  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Tests: error paths
# ---------------------------------------------------------------------------


class TestResolveContractErrors:
    """Error path coverage for resolve_contract."""

    def test_mesh_raises_contract_inference_error(self, mesh_geometry: GeometrySpec) -> None:
        """MESH geometry raises ContractInferenceError with helpful message.

        Args:
            mesh_geometry: Unsupported mesh geometry fixture.
        """
        with pytest.raises(ContractInferenceError, match="MESH"):
            resolve_contract(mesh_geometry)

    def test_no_feature_field_raises_value_error(self, no_feature_geometry: GeometrySpec) -> None:
        """Geometry with no FEATURE field raises ValueError from primary_feature().

        Args:
            no_feature_geometry: Geometry missing a FEATURE-role field.
        """
        with pytest.raises(ValueError, match="no FEATURE field"):
            resolve_contract(no_feature_geometry)
