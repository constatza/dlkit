"""Tests for the pluggable graph-convolution Strategy layer (``graph/conv.py``)."""

from __future__ import annotations

import inspect

import pytest
import torch

from dlkit.domain.nn.graph.conv import (
    GATv2Message,
    GraphConvKind,
    GraphMessage,
    SimpleGATv2Message,
    SimpleGraphMessage,
    _GATv2ConvAdapter,
    _GCNConvAdapter,
    _SAGEConvAdapter,
    resolve_graph_conv_factory,
)


def test_resolve_graph_conv_factory_rejects_unsupported_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported graph conv kind"):
        resolve_graph_conv_factory("unknown")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("conv_kind", ["gatv2", "gcn", "sage"])
def test_resolve_graph_conv_factory_builds_module_of_requested_shape(
    conv_kind: GraphConvKind, node_features: torch.Tensor, edge_index: torch.Tensor
) -> None:
    factory = resolve_graph_conv_factory(conv_kind, residual=False)
    module = factory(node_features.shape[-1], 4)
    out = module(node_features, edge_index)
    assert out.shape == (node_features.shape[0], 4)


class TestGATv2ConvAdapter:
    def test_residual_true_builds_native_res_projection(self) -> None:
        adapter = _GATv2ConvAdapter(3, 4, heads=1, residual=True)
        assert adapter.res is not None

    def test_residual_false_has_no_res_projection(self) -> None:
        adapter = _GATv2ConvAdapter(3, 4, heads=1, residual=False)
        assert adapter.res is None

    def test_concat_requires_out_channels_divisible_by_heads(self) -> None:
        with pytest.raises(ValueError, match="divisible by heads"):
            _GATv2ConvAdapter(3, 5, heads=2, concat=True)

    def test_forward_passes_edge_attr_through(
        self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr_1d: torch.Tensor
    ) -> None:
        adapter = _GATv2ConvAdapter(3, 4, heads=1, edge_dim=1)
        out = adapter(node_features, edge_index, edge_attr_1d)
        assert out.shape == (3, 4)


class TestGCNConvAdapter:
    @pytest.mark.parametrize("edge_dim", [None, 1])
    def test_valid_edge_dim_constructs(self, edge_dim: int | None) -> None:
        _GCNConvAdapter(3, 4, edge_dim=edge_dim)

    def test_invalid_edge_dim_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match="edge_dim must be None or 1"):
            _GCNConvAdapter(3, 4, edge_dim=2)

    def test_forward_squeezes_edge_attr_into_edge_weight(
        self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr_1d: torch.Tensor
    ) -> None:
        adapter = _GCNConvAdapter(3, 4, edge_dim=1)
        out = adapter(node_features, edge_index, edge_attr_1d)
        assert out.shape == (3, 4)

    def test_residual_adds_input_to_conv_output(
        self, node_features: torch.Tensor, edge_index: torch.Tensor
    ) -> None:
        torch.manual_seed(0)
        plain = _GCNConvAdapter(3, 3, residual=False)
        torch.manual_seed(0)
        residual = _GCNConvAdapter(3, 3, residual=True)

        plain_out = plain(node_features, edge_index)
        residual_out = residual(node_features, edge_index)

        assert torch.allclose(residual_out, node_features + plain_out, atol=1e-6)


class TestSAGEConvAdapter:
    def test_none_edge_dim_constructs(self) -> None:
        _SAGEConvAdapter(3, 4, edge_dim=None)

    def test_non_none_edge_dim_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match="edge_dim must be None"):
            _SAGEConvAdapter(3, 4, edge_dim=1)

    def test_forward_rejects_edge_attr_as_backstop(
        self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr_1d: torch.Tensor
    ) -> None:
        adapter = _SAGEConvAdapter(3, 4)
        with pytest.raises(ValueError, match="does not support edge features"):
            adapter(node_features, edge_index, edge_attr_1d)

    def test_residual_adds_input_to_conv_output(
        self, node_features: torch.Tensor, edge_index: torch.Tensor
    ) -> None:
        torch.manual_seed(0)
        plain = _SAGEConvAdapter(3, 3, residual=False)
        torch.manual_seed(0)
        residual = _SAGEConvAdapter(3, 3, residual=True)

        plain_out = plain(node_features, edge_index)
        residual_out = residual(node_features, edge_index)

        assert torch.allclose(residual_out, node_features + plain_out, atol=1e-6)


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


def test_gatv2_message_passes_residual_true_to_conv() -> None:
    module = GATv2Message(hidden_size=8, num_layers=1, heads=1)
    assert module.layers[0].res is not None


def test_simple_gatv2_message_passes_residual_false_to_conv() -> None:
    module = SimpleGATv2Message(hidden_size=8, num_layers=1, heads=1)
    assert module.layers[0].res is None


def test_gatv2_message_is_conv_kind_gatv2_preset() -> None:
    assert issubclass(GATv2Message, GraphMessage)
    assert issubclass(SimpleGATv2Message, SimpleGraphMessage)
