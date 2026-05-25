"""Geometry contracts for DLKit — physics-role and spatial-structure enumerations.

These value objects are shared across all layers and must not import from any
layer above ``common``.
"""

from dataclasses import dataclass
from enum import StrEnum


class FieldRole(StrEnum):
    """Physics semantics — what the field represents in the problem domain."""

    FEATURE = "feature"
    FEATURE_COORDINATES = "feature_coordinates"
    TARGET_COORDINATES = "target_coordinates"
    # EDGE_FEATURE deferred


class GeometryKind(StrEnum):
    """Spatial structure of the data."""

    TABULAR = "tabular"
    REGULAR_GRID = "regular_grid"
    SEQUENCE = "sequence"
    POINT_CLOUD = "point_cloud"
    GRAPH = "graph"
    MESH = "mesh"


class TopologyKind(StrEnum):
    """Encoding used to represent connectivity information."""

    EDGE_INDEX = "edge_index"
    CELL_COMPLEX = "cell_complex"


@dataclass(frozen=True, slots=True, kw_only=True)
class FieldSpec:
    """Specification for a single named field in a dataset entry.

    Attributes:
        name: Identifier for this field.
        shape: Tensor shape excluding the batch dimension.
        role: Physics role this field plays in the problem.
        geometry_kind: Spatial structure of the field data.
    """

    name: str
    shape: tuple[int, ...]
    role: FieldRole
    geometry_kind: GeometryKind = GeometryKind.TABULAR

    def __post_init__(self) -> None:
        """Validates that shape is non-empty.

        Raises:
            ValueError: If shape is an empty tuple.
        """
        if not self.shape:
            raise ValueError("FieldSpec.shape must have at least one dimension")

    @property
    def primary_size(self) -> int:
        """Returns the leading (first) dimension of the field shape.

        Returns:
            The value of ``shape[0]``.
        """
        return self.shape[0]


@dataclass(frozen=True, slots=True, kw_only=True)
class GeometrySpec:
    """Aggregate geometry contract for a dataset or model interface.

    Attributes:
        fields: Ordered tuple of :class:`FieldSpec` instances, following
            ``entry_configs`` insertion order.
        topology_kind: Optional connectivity encoding (e.g. edge index).
        edge_feature_dim: Optional dimensionality of edge features.
    """

    fields: tuple[FieldSpec, ...]
    topology_kind: TopologyKind | None = None
    edge_feature_dim: int | None = None

    def by_role(self, role: FieldRole) -> tuple[FieldSpec, ...]:
        """Filters fields by their :class:`FieldRole`.

        Args:
            role: The role to filter on.

        Returns:
            A tuple of all :class:`FieldSpec` instances with the given role,
            preserving insertion order.
        """
        return tuple(f for f in self.fields if f.role == role)

    def primary_feature(self) -> FieldSpec:
        """Returns the first FEATURE-role field.

        Returns:
            The first :class:`FieldSpec` with role :attr:`FieldRole.FEATURE`.

        Raises:
            ValueError: If no FEATURE-role field is present.
        """
        features = self.by_role(FieldRole.FEATURE)
        if not features:
            raise ValueError("GeometrySpec has no FEATURE field")
        return features[0]
