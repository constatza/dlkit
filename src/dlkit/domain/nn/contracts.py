"""Lean shape-bundle contracts for model construction.

Each contract variant carries only the shape information a model family needs.
No geometry, no config — just the minimal data required to build the model.
"""

from __future__ import annotations

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


# Sum type using PEP 695 syntax
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


__all__ = [
    "BranchTrunkSpec",
    "ContractConsumer",
    "GraphContractSpec",
    "GridOperatorSpec",
    "ModelContractSpec",
    "SequenceSpec",
    "TabulaRSpec",
]
