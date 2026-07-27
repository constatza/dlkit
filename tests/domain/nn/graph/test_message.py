"""Tests for stacked graph message-passing modules (``graph/message.py``)."""

from __future__ import annotations

import inspect

import pytest
import torch

from dlkit.domain.nn.graph.conv import GraphConvKind
from dlkit.domain.nn.graph.message import (
    GATv2Message,
    GraphMessage,
    SimpleGATv2Message,
    SimpleGraphMessage,
)


def _param_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@pytest.mark.parametrize("conv_kind", ["gatv2", "gcn", "sage"])
def test_graph_message_forward_shape_across_conv_kinds(
    conv_kind: GraphConvKind, node_features: torch.Tensor, edge_index: torch.Tensor
) -> None:
    module = GraphMessage(conv_kind=conv_kind, hidden_size=3, num_layers=2)
    out = module(node_features, edge_index)
    assert out.shape == node_features.shape


def test_graph_message_has_no_public_residual_param() -> None:
    assert "residual" not in inspect.signature(GraphMessage.__init__).parameters


def test_simple_graph_message_has_no_public_residual_param() -> None:
    assert "residual" not in inspect.signature(SimpleGraphMessage.__init__).parameters


def test_gatv2_message_has_no_public_residual_param() -> None:
    assert "residual" not in inspect.signature(GATv2Message.__init__).parameters


def test_simple_gatv2_message_has_no_public_residual_param() -> None:
    assert "residual" not in inspect.signature(SimpleGATv2Message.__init__).parameters


def test_gatv2_message_residual_adds_a_learned_residual_projection() -> None:
    plain = SimpleGATv2Message(hidden_size=8, num_layers=1, heads=1)
    residual = GATv2Message(hidden_size=8, num_layers=1, heads=1)
    assert _param_count(residual) > _param_count(plain)


def test_gatv2_message_is_conv_kind_gatv2_preset() -> None:
    assert issubclass(GATv2Message, GraphMessage)
    assert issubclass(SimpleGATv2Message, SimpleGraphMessage)
