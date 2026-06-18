"""Unit tests for the public surface of ``dlkit.domain.nn.contracts``.

The contracts module now exposes only ``EntryConsumer`` (a runtime-checkable
protocol) and ``InputSpec`` (a permissive pydantic base). These tests cover the
structural protocol check and the ``InputSpec`` extra-field behaviour.
"""

from __future__ import annotations

from typing import Any, Self

import pytest
import torch.nn as nn
from pydantic import BaseModel

from dlkit.common.shapes import InputShapes, OutputShapes
from dlkit.domain.nn.contracts import EntryConsumer, InputSpec
from dlkit.domain.nn.ffnn.residual import FFNN

# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------


class _EntryConsumerModel(nn.Module):
    """nn.Module that structurally satisfies ``EntryConsumer``."""

    InputSpec: type[InputSpec] = InputSpec

    @classmethod
    def from_entries(
        cls,
        input_shapes: InputShapes,
        output_shapes: OutputShapes,
        **kwargs: Any,
    ) -> Self:
        return cls()

    def forward(self, x: Any) -> Any:  # noqa: ANN401
        return x


class _NoEntriesModel(nn.Module):
    """nn.Module lacking both ``from_entries`` and ``InputSpec``."""

    def forward(self, x: Any) -> Any:  # noqa: ANN401
        return x


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def entry_consumer_cls() -> type[_EntryConsumerModel]:
    """A model class that satisfies the EntryConsumer protocol."""
    return _EntryConsumerModel


@pytest.fixture
def non_consumer_cls() -> type[_NoEntriesModel]:
    """A model class that does not satisfy the EntryConsumer protocol."""
    return _NoEntriesModel


@pytest.fixture
def input_shapes() -> InputShapes:
    """Single named feature shape."""
    return {"x": (4,)}


@pytest.fixture
def output_shapes() -> OutputShapes:
    """Single named target shape."""
    return {"y": (2,)}


@pytest.fixture
def input_spec_payload() -> dict[str, Any]:
    """Arbitrary keyword payload for InputSpec extra-field tests."""
    return {"x": (4,), "extra_entry": (8, 8)}


# ---------------------------------------------------------------------------
# EntryConsumer protocol
# ---------------------------------------------------------------------------


class TestEntryConsumerProtocol:
    def test_is_runtime_checkable(self) -> None:
        """EntryConsumer supports isinstance/issubclass at runtime."""
        assert hasattr(EntryConsumer, "_is_runtime_protocol")

    def test_structural_consumer_satisfies_protocol(
        self, entry_consumer_cls: type[_EntryConsumerModel]
    ) -> None:
        """A class declaring from_entries + InputSpec satisfies the protocol."""
        assert isinstance(entry_consumer_cls, EntryConsumer)

    def test_non_consumer_does_not_satisfy_protocol(
        self, non_consumer_cls: type[_NoEntriesModel]
    ) -> None:
        """A class missing the protocol members is rejected."""
        assert not isinstance(non_consumer_cls, EntryConsumer)

    def test_from_entries_constructs_instance(
        self,
        entry_consumer_cls: type[_EntryConsumerModel],
        input_shapes: InputShapes,
        output_shapes: OutputShapes,
    ) -> None:
        """from_entries builds an instance of the consumer class."""
        model = entry_consumer_cls.from_entries(input_shapes, output_shapes)
        assert isinstance(model, entry_consumer_cls)

    def test_real_model_exposes_from_entries_classmethod(self) -> None:
        """Real DLKit models expose a from_entries classmethod."""
        assert callable(FFNN.from_entries)

    def test_real_model_builds_from_entries(
        self, input_shapes: InputShapes, output_shapes: OutputShapes
    ) -> None:
        """A real model builds with correct in/out dims from entry shapes."""
        model = FFNN.from_entries(input_shapes, output_shapes, hidden_size=8, num_layers=2)
        assert isinstance(model, FFNN)


# ---------------------------------------------------------------------------
# InputSpec
# ---------------------------------------------------------------------------


class TestInputSpec:
    def test_is_pydantic_base_model(self) -> None:
        """InputSpec is a pydantic BaseModel subclass."""
        assert issubclass(InputSpec, BaseModel)

    def test_allows_extra_fields(self, input_spec_payload: dict[str, Any]) -> None:
        """Extra (unmodelled) fields are permitted."""
        spec = InputSpec(**input_spec_payload)
        for key, value in input_spec_payload.items():
            assert getattr(spec, key) == value

    def test_empty_construction_is_valid(self) -> None:
        """InputSpec can be constructed with no fields."""
        assert isinstance(InputSpec(), InputSpec)
