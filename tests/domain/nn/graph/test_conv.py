"""Tests for the graph-convolution adapter/factory layer (``graph/conv.py``)."""

from __future__ import annotations

import pytest
import torch

from dlkit.domain.nn.graph.conv import (
    GraphConvKind,
    _GATv2ConvAdapter,
    _GCNConvAdapter,
    _SAGEConvAdapter,
    resolve_graph_conv_factory,
)


def _param_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


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


def test_resolve_graph_conv_factory_rejects_kwarg_from_a_different_conv_kind() -> None:
    """GCN has no `heads` knob (that's GATv2-specific) -- must fail as a clear TypeError
    naming the adapter, not an opaque error from deep inside PyTorch Geometric."""
    factory = resolve_graph_conv_factory("gcn", heads=4)
    with pytest.raises(TypeError, match="_GCNConvAdapter"):
        factory(3, 4)


class TestGATv2ConvAdapter:
    def test_residual_true_adds_a_learned_residual_projection(self) -> None:
        plain = _GATv2ConvAdapter(3, 4, heads=1, residual=False)
        residual = _GATv2ConvAdapter(3, 4, heads=1, residual=True)
        assert _param_count(residual) > _param_count(plain)

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
