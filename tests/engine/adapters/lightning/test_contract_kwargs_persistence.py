"""Tests for entry-shape serialization and checkpoint persistence."""

from __future__ import annotations

from typing import Any, Self
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from dlkit.common.sources import InputShapes, OutputShapes
from dlkit.engine.adapters.lightning.concerns._checkpoint_serializer_helpers import (
    deserialize_shapes,
    serialize_shapes,
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


class _DummyModel(nn.Module):
    """Minimal entry-consumer model referenced by model_settings."""

    @classmethod
    def from_entries(
        cls, input_shapes: InputShapes, output_shapes: OutputShapes, **kwargs: Any
    ) -> Self:
        return cls()

    def forward(self, x: Any) -> Any:  # noqa: ANN401
        return x


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def input_shapes() -> dict[str, tuple[int, ...]]:
    """Feature-name-to-shape mapping for testing."""
    return {"x": (4,)}


@pytest.fixture
def output_shapes() -> dict[str, tuple[int, ...]]:
    """Target-name-to-shape mapping for testing."""
    return {"y": (2,)}


@pytest.fixture
def grid_input_shapes() -> dict[str, tuple[int, ...]]:
    """Multi-dimensional feature shape for round-trip testing."""
    return {"u": (3, 8, 8)}


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
    return (ValueEntry(name="x", value=torch.zeros(4, 1), data_role=DataRole.FEATURE),)


@pytest.fixture
def metadata_with_shapes(
    model_settings, wrapper_settings, entry_configs, input_shapes, output_shapes
) -> WrapperCheckpointMetadata:
    """WrapperCheckpointMetadata carrying input/output shapes."""
    return WrapperCheckpointMetadata(
        model_settings=model_settings,
        wrapper_settings=wrapper_settings,
        entry_configs=entry_configs,
        predict_target_key="y",
        output_spec=ModelOutputSpec(),
        input_shapes=input_shapes,
        output_shapes=output_shapes,
    )


@pytest.fixture
def serializer_with_shapes(metadata_with_shapes) -> DLKitCheckpointSerializer:
    """DLKitCheckpointSerializer backed by metadata that carries shapes."""
    return DLKitCheckpointSerializer(metadata_with_shapes, model=MagicMock())


@pytest.fixture
def serializer_without_shapes(
    model_settings, wrapper_settings, entry_configs
) -> DLKitCheckpointSerializer:
    """DLKitCheckpointSerializer backed by metadata with no shapes."""
    meta = WrapperCheckpointMetadata(
        model_settings=model_settings,
        wrapper_settings=wrapper_settings,
        entry_configs=entry_configs,
        predict_target_key="y",
        output_spec=ModelOutputSpec(),
        input_shapes=None,
        output_shapes=None,
    )
    return DLKitCheckpointSerializer(meta, model=MagicMock())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_serialize_shapes_roundtrip(input_shapes, grid_input_shapes):
    """serialize_shapes/deserialize_shapes return equivalent shape mappings."""
    assert deserialize_shapes(serialize_shapes(input_shapes)) == dict(input_shapes)
    assert deserialize_shapes(serialize_shapes(grid_input_shapes)) == dict(grid_input_shapes)


def test_serialize_shapes_returns_none_for_empty():
    """serialize_shapes/deserialize_shapes return None for empty input."""
    assert serialize_shapes(None) is None
    assert serialize_shapes({}) is None
    assert deserialize_shapes(None) is None
    assert deserialize_shapes({}) is None


def test_build_checkpoint_metadata_stores_shapes(
    model_settings, wrapper_settings, entry_configs, input_shapes, output_shapes
):
    """build_checkpoint_metadata forwards shapes to WrapperCheckpointMetadata."""
    meta = build_checkpoint_metadata(
        model_settings=model_settings,
        wrapper_settings=wrapper_settings,
        entry_configs=entry_configs,
        predict_target_key="y",
        output_spec=ModelOutputSpec(),
        input_shapes=input_shapes,
        output_shapes=output_shapes,
    )
    assert meta.input_shapes == input_shapes
    assert meta.output_shapes == output_shapes


def test_serialize_stores_shapes_in_dlkit_metadata(
    serializer_with_shapes, input_shapes, output_shapes
):
    """serialize() writes shapes under dlkit_metadata input/output keys."""
    checkpoint: dict = {}
    serializer_with_shapes.serialize(checkpoint, "StandardLightningWrapper")

    assert checkpoint["dlkit_metadata"]["input_shapes"] == serialize_shapes(input_shapes)
    assert checkpoint["dlkit_metadata"]["output_shapes"] == serialize_shapes(output_shapes)


def test_no_shapes_stores_null_in_metadata(serializer_without_shapes):
    """serialize() writes None for shapes when no shapes were supplied."""
    checkpoint: dict = {}
    serializer_without_shapes.serialize(checkpoint, "StandardLightningWrapper")

    assert checkpoint["dlkit_metadata"]["input_shapes"] is None
    assert checkpoint["dlkit_metadata"]["output_shapes"] is None
