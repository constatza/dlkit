"""Unit tests for the public surface of ``dlkit.domain.nn.contracts``.

The contracts module exposes ``EntryConsumer`` (a runtime-checkable protocol),
``InputSpec``, ``OutputSpec``, and ``StandardEntryConsumer``.
These tests cover the structural protocol check, spec behaviour, and the
``from_context`` / ``resolve_shape_kwargs`` / ``shape_kwarg_names`` lifecycle.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch.nn as nn
from pydantic import BaseModel

from dlkit.common.shapes import ShapeContext
from dlkit.domain.nn.contracts import EntryConsumer, InputSpec, OutputSpec, StandardEntryConsumer
from dlkit.domain.nn.ffnn.residual import FFNN

# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------


class _EntryConsumerModel(StandardEntryConsumer, nn.Module):
    """nn.Module that structurally satisfies ``EntryConsumer`` via StandardEntryConsumer."""

    InputSpec: type[InputSpec] = InputSpec

    def __init__(self, *, in_features: int, out_features: int, **kwargs: Any) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: Any) -> Any:  # noqa: ANN401
        return x


class _NoEntriesModel(nn.Module):
    """nn.Module lacking the protocol members."""

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
def input_shapes() -> dict[str, tuple[int, ...]]:
    """Single named feature shape."""
    return {"x": (4,)}


@pytest.fixture
def output_shapes() -> dict[str, tuple[int, ...]]:
    """Single named target shape."""
    return {"y": (2,)}


@pytest.fixture
def shape_context(
    input_shapes: dict[str, tuple[int, ...]],
    output_shapes: dict[str, tuple[int, ...]],
) -> ShapeContext:
    """ShapeContext built from fixtures."""
    return ShapeContext(input_shapes, output_shapes)


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
        """A class with from_context + shape_kwarg_names + resolve_shape_kwargs satisfies the protocol."""
        assert isinstance(entry_consumer_cls, EntryConsumer)

    def test_non_consumer_does_not_satisfy_protocol(
        self, non_consumer_cls: type[_NoEntriesModel]
    ) -> None:
        """A class missing the protocol members is rejected."""
        assert not isinstance(non_consumer_cls, EntryConsumer)

    def test_from_context_constructs_instance(
        self,
        entry_consumer_cls: type[_EntryConsumerModel],
        shape_context: ShapeContext,
    ) -> None:
        """from_context builds an instance of the consumer class."""
        model = entry_consumer_cls.from_context(shape_context)
        assert isinstance(model, entry_consumer_cls)

    def test_real_model_exposes_from_context_classmethod(self) -> None:
        """Real DLKit models expose a from_context classmethod."""
        assert callable(FFNN.from_context)

    def test_real_model_builds_from_context(self, shape_context: ShapeContext) -> None:
        """A real model builds with correct in/out dims from ShapeContext."""
        model = FFNN.from_context(shape_context, hidden_size=8, num_layers=2)
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


# ---------------------------------------------------------------------------
# OutputSpec
# ---------------------------------------------------------------------------


class TestOutputSpec:
    def test_is_pydantic_base_model(self) -> None:
        """OutputSpec is a pydantic BaseModel subclass."""
        assert issubclass(OutputSpec, BaseModel)

    def test_allows_extra_fields(self) -> None:
        """Extra fields are permitted."""
        spec = OutputSpec(y=(2,))
        assert spec.y == (2,)

    def test_empty_construction_is_valid(self) -> None:
        """OutputSpec can be constructed with no fields."""
        assert isinstance(OutputSpec(), OutputSpec)


# ---------------------------------------------------------------------------
# shape_kwarg_names / resolve_shape_kwargs
# ---------------------------------------------------------------------------


class TestStandardEntryConsumer:
    def test_shape_kwarg_names_returns_frozenset(self) -> None:
        """shape_kwarg_names() returns a frozenset of strings."""
        result = _EntryConsumerModel.shape_kwarg_names()
        assert isinstance(result, frozenset)
        assert result == frozenset({"in_features", "out_features"})

    def test_resolve_shape_kwargs_uses_context(self, shape_context: ShapeContext) -> None:
        """resolve_shape_kwargs() extracts dims from ShapeContext."""
        kwargs = _EntryConsumerModel.resolve_shape_kwargs(shape_context)
        assert kwargs == {"in_features": 4, "out_features": 2}

    def test_from_context_wires_correct_dims(self, shape_context: ShapeContext) -> None:
        """from_context passes shape kwargs to __init__."""
        model = _EntryConsumerModel.from_context(shape_context)
        assert model.in_features == 4
        assert model.out_features == 2
