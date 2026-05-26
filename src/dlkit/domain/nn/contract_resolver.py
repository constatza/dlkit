"""Pure function for lifting a GeometrySpec to a ModelContractSpec variant.

Lives in ``domain.nn`` so that both ``engine.inference`` and
``engine.workflows.factories`` can import it without violating the package DAG.
Only depends on ``common.geometry`` and ``domain.nn.contracts``.
"""

from __future__ import annotations

from dlkit.common.errors import DLKitError
from dlkit.common.geometry import FieldRole, GeometryKind, GeometrySpec

from .contracts import (
    BranchTrunkSpec,
    GraphContractSpec,
    GridOperatorSpec,
    ModelContractSpec,
    SequenceSpec,
    TabulaRSpec,
)


class ContractInferenceError(DLKitError):
    """Raised when a GeometrySpec cannot be mapped to a ModelContractSpec."""


def resolve_contract(
    geometry: GeometrySpec,
    output_shapes: tuple[tuple[int, ...], ...] = (),
) -> ModelContractSpec:
    """Lift a GeometrySpec to the appropriate ModelContractSpec variant.

    Dispatches on the primary feature field's GeometryKind. This is the
    OCP extension point: add a new GeometryKind → add one new match arm.

    Args:
        geometry: Input field geometry (feature fields only).
        output_shapes: Shapes of target/output fields, in config order.
                       First element is used as the primary output shape.

    Returns:
        The matching ModelContractSpec variant.

    Raises:
        ContractInferenceError: If geometry is unsupported or ambiguous.
        ValueError: Propagated from geometry.primary_feature() if no FEATURE field.
    """
    primary = geometry.primary_feature()

    match primary.geometry_kind:
        case GeometryKind.GRAPH:
            return _build_graph_contract(geometry)
        case GeometryKind.SEQUENCE:
            return _build_sequence_contract(geometry, output_shapes)
        case GeometryKind.REGULAR_GRID | GeometryKind.POINT_CLOUD:
            if geometry.by_role(FieldRole.TARGET_COORDINATES):
                return _build_branch_trunk_contract(geometry, output_shapes)
            return _build_grid_contract(geometry, output_shapes)
        case GeometryKind.TABULAR:
            if geometry.by_role(FieldRole.TARGET_COORDINATES):
                return _build_branch_trunk_contract(geometry, output_shapes)
            return _build_tabular_contract(geometry, output_shapes)
        case GeometryKind.MESH:
            raise ContractInferenceError("MESH geometry is not yet supported")


# ---------------------------------------------------------------------------
# Private builder helpers
# ---------------------------------------------------------------------------


def _build_tabular_contract(
    geometry: GeometrySpec,
    output_shapes: tuple[tuple[int, ...], ...],
) -> TabulaRSpec:
    """Build a TabulaRSpec from tabular geometry.

    Args:
        geometry: Input geometry containing FEATURE fields.
        output_shapes: Target output shapes; first element used as out_shape.

    Returns:
        TabulaRSpec with in_shape and out_shape.
    """
    primary = geometry.primary_feature()
    if not output_shapes:
        from dlkit.common.errors import WorkflowError

        raise WorkflowError(
            "Cannot resolve tabular output shape: checkpoint predates contract persistence. "
            "Retrain the model to generate a compatible checkpoint."
        )
    out = output_shapes[0]
    return TabulaRSpec(in_shape=primary.shape, out_shape=out)


def _build_grid_contract(
    geometry: GeometrySpec,
    output_shapes: tuple[tuple[int, ...], ...],
) -> GridOperatorSpec:
    """Build a GridOperatorSpec from grid/point-cloud geometry.

    Args:
        geometry: Input geometry containing FEATURE fields.
        output_shapes: Target output shapes; first element's first dim used as out_channels.

    Returns:
        GridOperatorSpec with in_channels, out_channels, and spatial_shape.
    """
    primary = geometry.primary_feature()
    out = output_shapes[0] if output_shapes else (1,)
    return GridOperatorSpec(
        in_channels=primary.primary_size,
        out_channels=out[0],
        spatial_shape=primary.shape[1:],
    )


def _build_sequence_contract(
    geometry: GeometrySpec,
    output_shapes: tuple[tuple[int, ...], ...],
) -> SequenceSpec:
    """Build a SequenceSpec from sequence geometry.

    Args:
        geometry: Input geometry containing FEATURE fields.
        output_shapes: Target output shapes; first element used to derive out_channels.

    Returns:
        SequenceSpec with in_channels, seq_len, and out_channels.
    """
    primary = geometry.primary_feature()
    out = output_shapes[0] if output_shapes else (1,)
    return SequenceSpec(
        in_channels=primary.primary_size,
        seq_len=primary.shape[1],
        out_channels=out[0],
    )


def _build_branch_trunk_contract(
    geometry: GeometrySpec,
    output_shapes: tuple[tuple[int, ...], ...],
) -> BranchTrunkSpec:
    """Build a BranchTrunkSpec from geometry with TARGET_COORDINATES fields.

    Args:
        geometry: Input geometry containing FEATURE and TARGET_COORDINATES fields.
        output_shapes: Target output shapes; first element's first dim used as out_features.

    Returns:
        BranchTrunkSpec with branch_shape, query_shape, and out_features.
    """
    branch = geometry.primary_feature()
    query_fields = geometry.by_role(FieldRole.TARGET_COORDINATES)
    if not query_fields:
        raise ContractInferenceError(
            "BranchTrunkSpec requires at least one TARGET_COORDINATES field"
        )
    query = query_fields[0]
    out = output_shapes[0] if output_shapes else (1,)
    return BranchTrunkSpec(
        branch_shape=branch.shape,
        query_shape=query.shape,
        out_features=out[0],
    )


def _build_graph_contract(geometry: GeometrySpec) -> GraphContractSpec:
    """Build a GraphContractSpec from graph geometry.

    Args:
        geometry: Input geometry containing FEATURE fields with GRAPH kind.

    Returns:
        GraphContractSpec with in_channels, out_channels, and optional edge_dim.
    """
    primary = geometry.primary_feature()
    return GraphContractSpec(
        in_channels=primary.primary_size,
        out_channels=primary.primary_size,
        edge_dim=geometry.edge_feature_dim,
    )


__all__ = [
    "ContractInferenceError",
    "resolve_contract",
]
