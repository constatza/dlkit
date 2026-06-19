"""Regression tests for DeepONet construction via the entry-shapes API."""

from __future__ import annotations

import pytest

from dlkit.common.shapes import InputShapes, OutputShapes, ShapeContext
from dlkit.domain.nn.factory import build_model
from dlkit.domain.nn.operators.deeponet import VarWidthDeepONet


@pytest.fixture
def deeponet_input_shapes() -> InputShapes:
    """Branch and query input shapes keyed by entry name."""
    return {"branch": (10,), "query": (2,)}


@pytest.fixture
def deeponet_output_shapes() -> OutputShapes:
    """Output shape keyed by target entry name."""
    return {"y": (1,)}


@pytest.fixture
def deeponet_kwargs() -> dict[str, list[int]]:
    """Branch/trunk layer widths for VarWidthDeepONet."""
    return {"branch_layers": [64], "trunk_layers": [64]}


def test_deeponet_exposes_from_entries() -> None:
    """VarWidthDeepONet exposes a from_entries classmethod for entry-shape build."""
    assert callable(VarWidthDeepONet.from_context)


def test_deeponet_build_via_factory(
    deeponet_input_shapes: InputShapes,
    deeponet_output_shapes: OutputShapes,
    deeponet_kwargs: dict[str, list[int]],
) -> None:
    """build_model constructs a VarWidthDeepONet from entry shapes."""
    model = build_model(
        VarWidthDeepONet,
        deeponet_kwargs,
        context=ShapeContext(deeponet_input_shapes, deeponet_output_shapes),
    )
    assert isinstance(model, VarWidthDeepONet)


def test_deeponet_from_entries_derives_dims(
    deeponet_input_shapes: InputShapes,
    deeponet_output_shapes: OutputShapes,
    deeponet_kwargs: dict[str, list[int]],
) -> None:
    """from_entries derives branch/query/out dimensions from the shapes."""
    model = VarWidthDeepONet.from_context(
        ShapeContext(deeponet_input_shapes, deeponet_output_shapes),
        **deeponet_kwargs,
    )
    assert model.out_features == deeponet_output_shapes["y"][0]
