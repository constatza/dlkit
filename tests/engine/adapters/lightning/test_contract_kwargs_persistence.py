"""Tests for contract serialization and checkpoint persistence."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dlkit.domain.nn.contracts import (
    GridOperatorSpec,
    TabulaRSpec,
    deserialize_contract,
    serialize_contract,
)
from dlkit.engine.adapters.lightning.concerns.checkpoint_serializer import DLKitCheckpointSerializer
from dlkit.engine.adapters.lightning.model_invoker import ModelOutputSpec
from dlkit.engine.adapters.lightning.wrapper_types import (
    WrapperCheckpointMetadata,
    build_checkpoint_metadata,
)
from dlkit.infrastructure.config import (
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.entry_types import ValueEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tabular_spec() -> TabulaRSpec:
    """Standard tabular contract spec for testing."""
    return TabulaRSpec(in_shape=(4,), out_shape=(2,))


@pytest.fixture
def grid_spec() -> GridOperatorSpec:
    """Standard grid operator contract spec for testing."""
    return GridOperatorSpec(in_channels=3, out_channels=1, spatial_shape=(8, 8))


@pytest.fixture
def model_settings() -> ModelComponentSettings:
    """Minimal model settings for testing."""
    return ModelComponentSettings(
        name="_DummyModel",
        module_path="tests.engine.adapters.lightning.test_contract_kwargs_persistence",
    )


@pytest.fixture
def wrapper_settings() -> WrapperComponentSettings:
    """Minimal wrapper settings for testing."""
    return WrapperComponentSettings()


@pytest.fixture
def entry_configs():
    """Minimal entry configurations with one feature."""
    import torch

    return (ValueEntry(name="x", value=torch.zeros(4, 1), data_role=DataRole.FEATURE),)


@pytest.fixture
def wrapper_checkpoint_metadata(
    model_settings, wrapper_settings, entry_configs, tabular_spec
) -> WrapperCheckpointMetadata:
    """WrapperCheckpointMetadata carrying a TabulaRSpec contract."""
    return WrapperCheckpointMetadata(
        model_settings=model_settings,
        wrapper_settings=wrapper_settings,
        entry_configs=entry_configs,
        predict_target_key="y",
        geometry=None,
        output_spec=ModelOutputSpec(),
        contract=tabular_spec,
    )


@pytest.fixture
def serializer_with_contract(wrapper_checkpoint_metadata) -> DLKitCheckpointSerializer:
    """DLKitCheckpointSerializer backed by metadata that carries a contract."""
    return DLKitCheckpointSerializer(wrapper_checkpoint_metadata, model=MagicMock())


@pytest.fixture
def serializer_without_contract(
    model_settings, wrapper_settings, entry_configs
) -> DLKitCheckpointSerializer:
    """DLKitCheckpointSerializer backed by metadata with contract=None."""
    meta = WrapperCheckpointMetadata(
        model_settings=model_settings,
        wrapper_settings=wrapper_settings,
        entry_configs=entry_configs,
        predict_target_key="y",
        geometry=None,
        output_spec=ModelOutputSpec(),
        contract=None,
    )
    return DLKitCheckpointSerializer(meta, model=MagicMock())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_serialize_contract_roundtrip(tabular_spec, grid_spec):
    """Serializing and deserializing a contract returns an equivalent spec."""
    restored_tabular = deserialize_contract(serialize_contract(tabular_spec))
    assert restored_tabular == tabular_spec

    restored_grid = deserialize_contract(serialize_contract(grid_spec))
    assert restored_grid == grid_spec


def test_deserialize_contract_returns_none_for_empty():
    """deserialize_contract returns None for empty or unrecognized data."""
    assert deserialize_contract({}) is None
    assert deserialize_contract({"_type": "Unknown"}) is None


def test_build_checkpoint_metadata_stores_contract(
    model_settings, wrapper_settings, entry_configs, tabular_spec
):
    """build_checkpoint_metadata forwards contract to WrapperCheckpointMetadata."""
    meta = build_checkpoint_metadata(
        model_settings=model_settings,
        wrapper_settings=wrapper_settings,
        entry_configs=entry_configs,
        predict_target_key="y",
        geometry=None,
        output_spec=ModelOutputSpec(),
        contract=tabular_spec,
    )
    assert meta.contract == tabular_spec


def test_serialize_stores_contract_in_dlkit_metadata(serializer_with_contract, tabular_spec):
    """serialize() writes contract under dlkit_metadata['contract']."""
    checkpoint: dict = {}
    serializer_with_contract.serialize(checkpoint, "StandardLightningWrapper")

    contract_data = checkpoint["dlkit_metadata"]["contract"]
    assert contract_data is not None
    assert contract_data["_type"] == "TabulaRSpec"
    assert tuple(contract_data["in_shape"]) == tabular_spec.in_shape
    assert tuple(contract_data["out_shape"]) == tabular_spec.out_shape


def test_no_contract_stores_null_in_metadata(serializer_without_contract):
    """serialize() writes None under dlkit_metadata['contract'] when no contract."""
    checkpoint: dict = {}
    serializer_without_contract.serialize(checkpoint, "StandardLightningWrapper")

    assert checkpoint["dlkit_metadata"]["contract"] is None
