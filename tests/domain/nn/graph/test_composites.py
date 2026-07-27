"""Tests for EmbeddedHyperGraphNetwork/EmbeddedMoEGraphNetwork."""

from __future__ import annotations

import inspect
from typing import Any, cast

import pytest
import torch

from dlkit.domain.nn.graph.composites import EmbeddedHyperGraphNetwork, EmbeddedMoEGraphNetwork
from dlkit.domain.nn.graph.conv import GraphConvKind
from dlkit.domain.nn.graph.projection_networks import GProjection
from dlkit.domain.nn.primitives import GraphHyperSequential

_CONV_KIND_CASES = [
    pytest.param("gatv2", 1, True, id="gatv2"),
    pytest.param("gcn", 1, True, id="gcn"),
    pytest.param("sage", None, False, id="sage"),
]


@pytest.mark.parametrize(("conv_kind", "edge_dim", "use_edge_attr"), _CONV_KIND_CASES)
def test_embedded_hyper_graph_network_forward_shape_across_conv_kinds(
    conv_kind: GraphConvKind,
    edge_dim: int | None,
    use_edge_attr: bool,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr_1d: torch.Tensor,
) -> None:
    module = EmbeddedHyperGraphNetwork(
        in_channels=3,
        out_channels=2,
        hidden_size=8,
        num_layers=2,
        conv_kind=conv_kind,
        edge_dim=edge_dim,
    )
    edge_attr = edge_attr_1d if use_edge_attr else None
    assert module(node_features, edge_index, edge_attr).shape == (3, 2)


def test_embedded_hyper_graph_network_is_gprojection_subclass() -> None:
    assert issubclass(EmbeddedHyperGraphNetwork, GProjection)


def test_embedded_hyper_graph_network_uses_graph_hyper_sequential() -> None:
    module = EmbeddedHyperGraphNetwork(in_channels=3, out_channels=2, hidden_size=8, num_layers=2)
    message_module = cast(Any, module)._message_module
    assert isinstance(message_module, GraphHyperSequential)


def test_embedded_hyper_graph_network_conv_layers_are_non_residual_gcn() -> None:
    """GCN/SAGE adapters expose ``residual`` as a plain public attribute."""
    module = EmbeddedHyperGraphNetwork(
        in_channels=3, out_channels=2, hidden_size=8, num_layers=2, conv_kind="gcn"
    )
    message_module = cast(GraphHyperSequential, cast(Any, module)._message_module)
    for hyper_layer in message_module.layers:
        wrapped = cast(Any, hyper_layer).module
        assert wrapped.conv.residual is False


def test_embedded_hyper_graph_network_conv_layers_are_non_residual_sage() -> None:
    module = EmbeddedHyperGraphNetwork(
        in_channels=3, out_channels=2, hidden_size=8, num_layers=2, conv_kind="sage"
    )
    message_module = cast(GraphHyperSequential, cast(Any, module)._message_module)
    for hyper_layer in message_module.layers:
        wrapped = cast(Any, hyper_layer).module
        assert wrapped.conv.residual is False


def test_embedded_hyper_graph_network_gatv2_conv_layers_are_non_residual_by_param_count() -> None:
    """GATv2's adapter has no public ``residual`` attribute (removed from the adapter),
    so compare parameter counts against a residual=True built factory instead, the
    same technique test_conv.py/test_message.py already use for this distinction.
    """
    from dlkit.domain.nn.graph.conv import resolve_graph_conv_factory

    non_residual_factory = resolve_graph_conv_factory("gatv2", residual=False)
    residual_factory = resolve_graph_conv_factory("gatv2", residual=True)
    non_residual_params = sum(p.numel() for p in non_residual_factory(8, 8).parameters())
    residual_params = sum(p.numel() for p in residual_factory(8, 8).parameters())

    module = EmbeddedHyperGraphNetwork(
        in_channels=3, out_channels=2, hidden_size=8, num_layers=1, conv_kind="gatv2"
    )
    message_module = cast(GraphHyperSequential, cast(Any, module)._message_module)
    wrapped = cast(Any, message_module.layers[0]).module
    wrapped_params = sum(p.numel() for p in wrapped.conv.parameters())

    assert wrapped_params == non_residual_params
    assert wrapped_params < residual_params


@pytest.mark.parametrize(("conv_kind", "edge_dim", "use_edge_attr"), _CONV_KIND_CASES)
def test_embedded_moe_graph_network_forward_shape_across_conv_kinds(
    conv_kind: GraphConvKind,
    edge_dim: int | None,
    use_edge_attr: bool,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr_1d: torch.Tensor,
) -> None:
    module = EmbeddedMoEGraphNetwork(
        in_channels=3,
        out_channels=2,
        hidden_size=8,
        num_layers=2,
        num_experts=3,
        conv_kind=conv_kind,
        edge_dim=edge_dim,
    )
    edge_attr = edge_attr_1d if use_edge_attr else None
    assert module(node_features, edge_index, edge_attr).shape == (3, 2)


def test_embedded_moe_graph_network_is_gprojection_subclass() -> None:
    assert issubclass(EmbeddedMoEGraphNetwork, GProjection)


def test_embedded_moe_graph_network_routing_sanity(
    node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr_1d: torch.Tensor
) -> None:
    module = EmbeddedMoEGraphNetwork(
        in_channels=3,
        out_channels=2,
        hidden_size=8,
        num_layers=2,
        num_experts=4,
        top_k=2,
        conv_kind="gatv2",
        edge_dim=1,
    )
    assert module(node_features, edge_index, edge_attr_1d).shape == (3, 2)


def test_embedded_moe_graph_network_supports_swiglu_block_kind(
    node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr_1d: torch.Tensor
) -> None:
    module = EmbeddedMoEGraphNetwork(
        in_channels=3,
        out_channels=2,
        hidden_size=8,
        num_layers=1,
        num_experts=2,
        block_kind="swiglu",
        conv_kind="gatv2",
        edge_dim=1,
    )
    assert module(node_features, edge_index, edge_attr_1d).shape == (3, 2)


def test_embedded_moe_graph_network_has_no_return_stats_parameter() -> None:
    sig = inspect.signature(EmbeddedMoEGraphNetwork.__init__)
    assert "return_stats" not in sig.parameters

    forward_sig = inspect.signature(EmbeddedMoEGraphNetwork.forward)
    assert "return_stats" not in forward_sig.parameters
