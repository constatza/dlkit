"""Unit tests for domain/nn/contracts.py and the contract path in factory.py."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, Self
from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from dlkit.domain.nn.contracts import (
    BranchTrunkSpec,
    ContractConsumer,
    GraphContractSpec,
    GridOperatorSpec,
    ModelContractSpec,
    SequenceSpec,
    TabulaRSpec,
)
from dlkit.domain.nn.factory import build_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tabular_spec() -> TabulaRSpec:
    """A basic TabulaRSpec."""
    return TabulaRSpec(in_shape=(16,), out_shape=(4,))


@pytest.fixture
def grid_spec() -> GridOperatorSpec:
    """A basic GridOperatorSpec."""
    return GridOperatorSpec(in_channels=3, out_channels=1, spatial_shape=(64, 64))


@pytest.fixture
def sequence_spec() -> SequenceSpec:
    """A basic SequenceSpec with default out_len."""
    return SequenceSpec(in_channels=8, seq_len=100, out_channels=4)


@pytest.fixture
def sequence_spec_with_horizon() -> SequenceSpec:
    """A SequenceSpec with explicit out_len."""
    return SequenceSpec(in_channels=8, seq_len=100, out_channels=4, out_len=10)


@pytest.fixture
def branch_trunk_spec() -> BranchTrunkSpec:
    """A basic BranchTrunkSpec."""
    return BranchTrunkSpec(branch_shape=(200,), query_shape=(2,), out_features=32)


@pytest.fixture
def graph_spec() -> GraphContractSpec:
    """A GraphContractSpec with edge features."""
    return GraphContractSpec(in_channels=16, out_channels=8, edge_dim=4)


@pytest.fixture
def graph_spec_no_edges() -> GraphContractSpec:
    """A GraphContractSpec without edge features."""
    return GraphContractSpec(in_channels=16, out_channels=8, edge_dim=None)


class _MinimalModule(nn.Module):
    """Minimal nn.Module that does nothing."""

    def forward(self, x: Any) -> Any:  # noqa: ANN401
        return x


class _ContractConsumerModel(_MinimalModule):
    """nn.Module that implements ContractConsumer."""

    received_contract: ModelContractSpec | None = None
    received_kwargs: dict[str, Any] | None = None

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        instance = cls()
        instance.received_contract = contract
        instance.received_kwargs = kwargs
        return instance


class _NoContractModel(_MinimalModule):
    """nn.Module that does NOT implement ContractConsumer."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden


# ---------------------------------------------------------------------------
# Frozen dataclass tests
# ---------------------------------------------------------------------------


class TestTabulaRSpec:
    def test_fields_are_accessible(self, tabular_spec: TabulaRSpec) -> None:
        assert tabular_spec.in_shape == (16,)
        assert tabular_spec.out_shape == (4,)

    def test_is_frozen(self, tabular_spec: TabulaRSpec) -> None:
        with pytest.raises(FrozenInstanceError):
            tabular_spec.in_shape = (32,)  # type: ignore[misc]


class TestGridOperatorSpec:
    def test_fields_are_accessible(self, grid_spec: GridOperatorSpec) -> None:
        assert grid_spec.in_channels == 3
        assert grid_spec.out_channels == 1
        assert grid_spec.spatial_shape == (64, 64)

    def test_is_frozen(self, grid_spec: GridOperatorSpec) -> None:
        with pytest.raises(FrozenInstanceError):
            grid_spec.in_channels = 99  # type: ignore[misc]


class TestSequenceSpec:
    def test_fields_are_accessible(self, sequence_spec: SequenceSpec) -> None:
        assert sequence_spec.in_channels == 8
        assert sequence_spec.seq_len == 100
        assert sequence_spec.out_channels == 4

    def test_default_out_len_is_none(self, sequence_spec: SequenceSpec) -> None:
        assert sequence_spec.out_len is None

    def test_explicit_out_len(self, sequence_spec_with_horizon: SequenceSpec) -> None:
        assert sequence_spec_with_horizon.out_len == 10

    def test_is_frozen(self, sequence_spec: SequenceSpec) -> None:
        with pytest.raises(FrozenInstanceError):
            sequence_spec.seq_len = 200  # type: ignore[misc]


class TestBranchTrunkSpec:
    def test_fields_are_accessible(self, branch_trunk_spec: BranchTrunkSpec) -> None:
        assert branch_trunk_spec.branch_shape == (200,)
        assert branch_trunk_spec.query_shape == (2,)
        assert branch_trunk_spec.out_features == 32

    def test_is_frozen(self, branch_trunk_spec: BranchTrunkSpec) -> None:
        with pytest.raises(FrozenInstanceError):
            branch_trunk_spec.out_features = 0  # type: ignore[misc]


class TestGraphContractSpec:
    def test_with_edge_dim(self, graph_spec: GraphContractSpec) -> None:
        assert graph_spec.edge_dim == 4

    def test_without_edge_dim(self, graph_spec_no_edges: GraphContractSpec) -> None:
        assert graph_spec_no_edges.edge_dim is None

    def test_is_frozen(self, graph_spec: GraphContractSpec) -> None:
        with pytest.raises(FrozenInstanceError):
            graph_spec.in_channels = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ContractConsumer protocol tests
# ---------------------------------------------------------------------------


class TestContractConsumerProtocol:
    def test_implementor_satisfies_protocol(self) -> None:
        assert issubclass(_ContractConsumerModel, ContractConsumer)

    def test_non_implementor_does_not_satisfy_protocol(self) -> None:
        assert not isinstance(_NoContractModel, ContractConsumer)

    def test_issubclass_is_correct_check_for_factory(self) -> None:
        """The factory dispatches on the class (type), not an instance.

        issubclass is the correct check; isinstance on the class also works for
        runtime_checkable protocols because the class itself is the checked object.
        """
        # issubclass is the correct check for factory usage.
        assert issubclass(_ContractConsumerModel, ContractConsumer)

    def test_missing_from_contract_fails_issubclass(self) -> None:
        assert not issubclass(_NoContractModel, ContractConsumer)


# ---------------------------------------------------------------------------
# build_model — contract path
# ---------------------------------------------------------------------------


class TestBuildModelContractPath:
    def test_calls_from_contract_when_consumer(self, tabular_spec: TabulaRSpec) -> None:
        model = build_model(
            _ContractConsumerModel,
            contract=tabular_spec,
        )
        assert isinstance(model, _ContractConsumerModel)
        assert model.received_contract is tabular_spec

    def test_forwards_kwargs_to_from_contract(self, tabular_spec: TabulaRSpec) -> None:
        model = build_model(
            _ContractConsumerModel,
            contract=tabular_spec,
            kwargs={"hidden": 128},
        )
        assert isinstance(model, _ContractConsumerModel)
        assert model.received_kwargs == {"hidden": 128}

    def test_no_contract_falls_through_to_kwargs(self) -> None:
        model = build_model(_NoContractModel, kwargs={"hidden": 32})
        assert isinstance(model, _NoContractModel)
        assert model.hidden == 32

    def test_none_contract_with_non_consumer_uses_kwargs(self) -> None:
        model = build_model(_NoContractModel, contract=None, kwargs={"hidden": 16})
        assert isinstance(model, _NoContractModel)
        assert model.hidden == 16

    def test_contract_ignored_for_non_consumer(self, tabular_spec: TabulaRSpec) -> None:
        """If model_cls doesn't implement ContractConsumer, contract is ignored."""
        model = build_model(
            _NoContractModel,
            contract=tabular_spec,
            kwargs={"hidden": 64},
        )
        assert isinstance(model, _NoContractModel)
        assert model.hidden == 64

    def test_uses_issubclass_not_isinstance(self, tabular_spec: TabulaRSpec) -> None:
        """Ensure the factory dispatches on the class (type), not an instance."""
        # Passing the class (not an instance) must work correctly
        model = build_model(_ContractConsumerModel, contract=tabular_spec)
        assert isinstance(model, _ContractConsumerModel)

    def test_empty_kwargs_defaults_to_empty_dict(self, tabular_spec: TabulaRSpec) -> None:
        model = build_model(_ContractConsumerModel, contract=tabular_spec)
        assert model.received_kwargs == {}


# ---------------------------------------------------------------------------
# build_model — legacy shape path still works
# ---------------------------------------------------------------------------


class _ShapeConsumerModel(_MinimalModule):
    """nn.Module that implements the legacy from_shape interface."""

    received_shape: Any = None

    @classmethod
    def from_shape(cls, shape: Any, **kwargs: Any) -> Self:
        instance = cls()
        instance.received_shape = shape
        return instance


class TestBuildModelShapePath:
    def test_legacy_from_shape_still_called(self) -> None:
        fake_shape = MagicMock()
        model = build_model(_ShapeConsumerModel, shape=fake_shape)
        assert isinstance(model, _ShapeConsumerModel)
        assert model.received_shape is fake_shape

    def test_shape_ignored_when_no_from_shape(self) -> None:
        fake_shape = MagicMock()
        model = build_model(_NoContractModel, shape=fake_shape, kwargs={"hidden": 8})
        assert isinstance(model, _NoContractModel)
        assert model.hidden == 8

    def test_contract_takes_priority_over_shape(self, tabular_spec: TabulaRSpec) -> None:
        """When both contract and shape are given, contract wins for ContractConsumer."""

        class _BothConsumer(_ContractConsumerModel, _ShapeConsumerModel):
            pass

        fake_shape = MagicMock()
        model = build_model(_BothConsumer, shape=fake_shape, contract=tabular_spec)
        assert isinstance(model, _BothConsumer)
        assert model.received_contract is tabular_spec
        # from_shape should NOT have been called
        assert model.received_shape is None
