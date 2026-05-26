from __future__ import annotations

import inspect
from typing import Any, cast

import torch

from dlkit.domain.nn.graph.gat import GATv2Message, SimpleGATv2Message
from dlkit.domain.nn.graph.projection_networks import GProjection
from dlkit.domain.nn.graph.scaled_projection_networks import (
    GATv2Projection,
    ScaledGATv2Projection,
    ScaledGProjection,
    ScaledSimpleGATv2Projection,
    SimpleGATv2Projection,
)


def _edge_index() -> torch.Tensor:
    return torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)


def _edge_attr() -> torch.Tensor:
    return torch.randn(_edge_index().shape[1], 1)


def test_gatv2_message_has_no_public_residual_param() -> None:
    sig = inspect.signature(GATv2Message.__init__)
    assert "residual" not in sig.parameters


def test_simple_gatv2_message_has_no_public_residual_param() -> None:
    sig = inspect.signature(SimpleGATv2Message.__init__)
    assert "residual" not in sig.parameters


def test_gatv2_message_passes_residual_true_to_conv() -> None:
    module = GATv2Message(hidden_size=8, num_layers=1, heads=1)
    assert module.layers[0].res is not None


def test_simple_gatv2_message_passes_residual_false_to_conv() -> None:
    module = SimpleGATv2Message(hidden_size=8, num_layers=1, heads=1)
    assert module.layers[0].res is None


def test_gatv2_projection_is_class_not_function() -> None:
    assert isinstance(GATv2Projection, type)


def test_gatv2_projection_is_gprojection_subclass() -> None:
    assert issubclass(GATv2Projection, GProjection)
    assert issubclass(SimpleGATv2Projection, GProjection)


def test_scaled_gatv2_projection_is_scaled_gprojection_subclass() -> None:
    assert issubclass(ScaledGATv2Projection, ScaledGProjection)
    assert issubclass(ScaledSimpleGATv2Projection, ScaledGProjection)


def test_gatv2_projection_uses_residual_message() -> None:
    module = GATv2Projection(in_channels=3, out_channels=2, hidden_size=8, num_layers=1)
    message_module = cast(Any, module)._message_module
    assert isinstance(message_module, GATv2Message)
    assert message_module.layers[0].res is not None


def test_simple_gatv2_projection_uses_plain_message() -> None:
    module = SimpleGATv2Projection(in_channels=3, out_channels=2, hidden_size=8, num_layers=1)
    message_module = cast(Any, module)._message_module
    assert isinstance(message_module, SimpleGATv2Message)
    assert message_module.layers[0].res is None


def test_scaled_simple_gatv2_projection_uses_plain_message() -> None:
    module = ScaledSimpleGATv2Projection(
        in_channels=3,
        out_channels=2,
        hidden_size=8,
        num_layers=1,
    )
    message_module = cast(Any, module)._message_module
    assert isinstance(message_module, SimpleGATv2Message)
    assert message_module.layers[0].res is None


def test_gatv2_projection_has_no_public_residual_param() -> None:
    sig = inspect.signature(GATv2Projection.__init__)
    assert "residual" not in sig.parameters


def test_graph_projection_variants_forward_shapes() -> None:
    x = torch.randn(3, 3)
    edge_index = _edge_index()
    edge_attr = _edge_attr()

    residual = GATv2Projection(in_channels=3, out_channels=2, hidden_size=8, num_layers=1)
    plain = SimpleGATv2Projection(in_channels=3, out_channels=2, hidden_size=8, num_layers=1)
    scaled = ScaledGATv2Projection(in_channels=3, out_channels=2, hidden_size=8, num_layers=1)
    scaled_plain = ScaledSimpleGATv2Projection(
        in_channels=3,
        out_channels=2,
        hidden_size=8,
        num_layers=1,
    )

    assert residual(x, edge_index, edge_attr).shape == (3, 2)
    assert plain(x, edge_index, edge_attr).shape == (3, 2)
    assert scaled(x, edge_index, edge_attr).shape == (3, 2)
    assert scaled_plain(x, edge_index, edge_attr).shape == (3, 2)
