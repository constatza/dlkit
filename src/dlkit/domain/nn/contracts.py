"""Lean shape-bundle contracts for model construction.

Each contract variant carries only the shape information a model family needs.
No geometry, no config — just the minimal data required to build the model.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, Self, runtime_checkable


@dataclass(frozen=True, slots=True, kw_only=True)
class TabulaRSpec:
    """Flat models: FFNN, Siren, FourierFeature, HashEncoding.

    Attributes:
        in_shape: Input tensor shape (excluding batch dim).
        out_shape: Output tensor shape (excluding batch dim).
    """

    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.in_shape or not self.out_shape:
            raise ValueError("TabulaRSpec shapes must be non-empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class GridOperatorSpec:
    """Spatially-structured operators: FNO, SkipCAE1d.

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        spatial_shape: Grid dimensions without batch or channel dims.
    """

    in_channels: int
    out_channels: int
    spatial_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("GridOperatorSpec channels must be positive")
        if not self.spatial_shape:
            raise ValueError("GridOperatorSpec.spatial_shape must be non-empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class SequenceSpec:
    """Timeseries and 1-D sequence models: RNN, TCN, Transformer, S4.

    Attributes:
        in_channels: Number of input channels (features per timestep).
        seq_len: Input sequence length.
        out_channels: Number of output channels (features per output step).
        out_len: Output horizon; ``None`` means same as ``seq_len``.
    """

    in_channels: int
    seq_len: int
    out_channels: int
    out_len: int | None = None

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.out_channels <= 0 or self.seq_len <= 0:
            raise ValueError("SequenceSpec int fields must be positive")
        if self.out_len is not None and self.out_len <= 0:
            raise ValueError("SequenceSpec.out_len must be positive when set")

    @property
    def effective_out_len(self) -> int:
        """Resolved output horizon; defaults to seq_len when out_len is None."""
        return self.out_len if self.out_len is not None else self.seq_len


@dataclass(frozen=True, slots=True, kw_only=True)
class BranchTrunkSpec:
    """DeepONet-style: branch sensor input + target coordinate trunk.

    Attributes:
        branch_shape: Shape of branch (sensor) input (excluding batch dim).
        query_shape: Shape of trunk (coordinate query) input (excluding batch dim).
        out_features: Number of output features.
    """

    branch_shape: tuple[int, ...]
    query_shape: tuple[int, ...]
    out_features: int

    def __post_init__(self) -> None:
        if not self.branch_shape or not self.query_shape:
            raise ValueError("BranchTrunkSpec shapes must be non-empty")
        if self.out_features <= 0:
            raise ValueError("BranchTrunkSpec.out_features must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphContractSpec:
    """GNN: node features, edge attributes, topology.

    Field names follow PyG conventions.

    Attributes:
        in_channels: Number of input node feature channels.
        out_channels: Number of output node feature channels.
        edge_dim: Edge feature dimensionality; ``None`` if no edge features.
    """

    in_channels: int
    out_channels: int
    edge_dim: int | None = None

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("GraphContractSpec channels must be positive")
        if self.edge_dim is not None and self.edge_dim <= 0:
            raise ValueError(
                "GraphContractSpec.edge_dim must be positive when set; use None for no edges"
            )


# Sum type — use match, not isinstance; ty enforces exhaustiveness per GeometryKind
type ModelContractSpec = (
    TabulaRSpec | GridOperatorSpec | SequenceSpec | BranchTrunkSpec | GraphContractSpec
)


@runtime_checkable
class ContractConsumer(Protocol):
    """Protocol for models that construct themselves from a ``ModelContractSpec``."""

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        """Construct the model from a contract spec.

        Args:
            contract: A ``ModelContractSpec`` variant matching this model family.
            **kwargs: Additional keyword arguments forwarded to the constructor.

        Returns:
            A fully constructed instance of this model.
        """
        ...


def serialize_contract(
    contract: ModelContractSpec,
) -> dict[str, int | tuple[int, ...] | None | str]:
    """Serialize a ModelContractSpec to a JSON-safe dict with type discriminator.

    Args:
        contract: Any ModelContractSpec variant.

    Returns:
        Dict with ``"_type"`` key for deserialization plus all contract fields.
    """
    return {"_type": type(contract).__name__, **dataclasses.asdict(contract)}


def deserialize_contract(
    data: Mapping[str, object],
) -> ModelContractSpec | None:
    """Reconstruct a ModelContractSpec from serialized checkpoint data.

    Handles list→tuple conversion for JSON round-trips.

    Args:
        data: Dict with ``"_type"`` key and contract field values.

    Returns:
        Reconstructed ModelContractSpec, or None if data is empty or unrecognized.
    """
    if not data:
        return None
    _REGISTRY: dict[str, type] = {
        cls.__name__: cls
        for cls in (TabulaRSpec, GridOperatorSpec, SequenceSpec, BranchTrunkSpec, GraphContractSpec)
    }
    cls = _REGISTRY.get(str(data.get("_type", "")))
    if cls is None:
        return None
    fields = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items() if k != "_type"}
    try:
        return cls(**fields)
    except Exception:
        return None


__all__ = [
    "BranchTrunkSpec",
    "ContractConsumer",
    "GraphContractSpec",
    "GridOperatorSpec",
    "ModelContractSpec",
    "SequenceSpec",
    "TabulaRSpec",
    "deserialize_contract",
    "serialize_contract",
]
