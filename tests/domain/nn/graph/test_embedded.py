"""Tests for EmbeddedGraphNetwork/ScaledEmbeddedGraphNetwork and the GATv2Projection presets."""

from __future__ import annotations

import inspect
from typing import Any, cast

import pytest
import torch

from dlkit.domain.nn.graph.conv import GraphConvKind, GraphMessage, SimpleGraphMessage
from dlkit.domain.nn.graph.embedded import EmbeddedGraphNetwork, ScaledEmbeddedGraphNetwork
from dlkit.domain.nn.graph.projection_networks import GProjection, ScaledGProjection
from dlkit.domain.nn.graph.scaled_projection_networks import (
    GATv2Projection,
    ScaledGATv2Projection,
    ScaledSimpleGATv2Projection,
    SimpleGATv2Projection,
)

_CONV_KIND_CASES = [
    pytest.param("gatv2", 1, True, id="gatv2"),
    pytest.param("gcn", 1, True, id="gcn"),
    pytest.param("sage", None, False, id="sage"),
]


@pytest.mark.parametrize(("conv_kind", "edge_dim", "use_edge_attr"), _CONV_KIND_CASES)
def test_embedded_graph_network_forward_shape_across_conv_kinds(
    conv_kind: GraphConvKind,
    edge_dim: int | None,
    use_edge_attr: bool,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr_1d: torch.Tensor,
) -> None:
    module = EmbeddedGraphNetwork(
        in_channels=3,
        out_channels=2,
        hidden_size=8,
        num_layers=1,
        conv_kind=conv_kind,
        edge_dim=edge_dim,
    )
    edge_attr = edge_attr_1d if use_edge_attr else None
    assert module(node_features, edge_index, edge_attr).shape == (3, 2)


@pytest.mark.parametrize(("conv_kind", "edge_dim", "use_edge_attr"), _CONV_KIND_CASES)
def test_scaled_embedded_graph_network_forward_shape_across_conv_kinds(
    conv_kind: GraphConvKind,
    edge_dim: int | None,
    use_edge_attr: bool,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr_1d: torch.Tensor,
) -> None:
    module = ScaledEmbeddedGraphNetwork(
        in_channels=3,
        out_channels=2,
        hidden_size=8,
        num_layers=1,
        conv_kind=conv_kind,
        edge_dim=edge_dim,
    )
    edge_attr = edge_attr_1d if use_edge_attr else None
    assert module(node_features, edge_index, edge_attr).shape == (3, 2)


def test_embedded_graph_network_is_gprojection_subclass() -> None:
    assert issubclass(EmbeddedGraphNetwork, GProjection)


def test_scaled_embedded_graph_network_is_scaled_gprojection_subclass() -> None:
    assert issubclass(ScaledEmbeddedGraphNetwork, ScaledGProjection)


def test_embedded_graph_network_residual_true_uses_graph_message() -> None:
    module = EmbeddedGraphNetwork(
        in_channels=3, out_channels=2, hidden_size=8, num_layers=1, residual=True
    )
    message_module = cast(Any, module)._message_module
    assert isinstance(message_module, GraphMessage)
    assert message_module.layers[0].res is not None


def test_embedded_graph_network_residual_false_uses_simple_graph_message() -> None:
    module = EmbeddedGraphNetwork(
        in_channels=3, out_channels=2, hidden_size=8, num_layers=1, residual=False
    )
    message_module = cast(Any, module)._message_module
    assert isinstance(message_module, SimpleGraphMessage)
    assert not isinstance(message_module, GraphMessage)
    assert message_module.layers[0].res is None


def test_embedded_graph_network_rejects_unsupported_conv_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported graph conv kind"):
        EmbeddedGraphNetwork(
            in_channels=3,
            out_channels=2,
            hidden_size=8,
            num_layers=1,
            conv_kind="unknown",  # ty: ignore[invalid-argument-type]
        )


def test_embedded_graph_network_gcn_conv_kind_rejects_invalid_edge_dim() -> None:
    with pytest.raises(ValueError, match="edge_dim must be None or 1"):
        EmbeddedGraphNetwork(
            in_channels=3, out_channels=2, hidden_size=8, num_layers=1, conv_kind="gcn", edge_dim=2
        )


def test_embedded_graph_network_sage_conv_kind_rejects_edge_dim() -> None:
    with pytest.raises(ValueError, match="edge_dim must be None"):
        EmbeddedGraphNetwork(
            in_channels=3, out_channels=2, hidden_size=8, num_layers=1, conv_kind="sage", edge_dim=1
        )


class TestGATv2ProjectionPresets:
    """The legacy GATv2Projection family, now thin presets over EmbeddedGraphNetwork."""

    def test_gatv2_projection_is_class_not_function(self) -> None:
        assert isinstance(GATv2Projection, type)

    def test_gatv2_projection_is_embedded_graph_network_subclass(self) -> None:
        assert issubclass(GATv2Projection, EmbeddedGraphNetwork)
        assert issubclass(SimpleGATv2Projection, EmbeddedGraphNetwork)

    def test_scaled_gatv2_projection_is_scaled_embedded_graph_network_subclass(self) -> None:
        assert issubclass(ScaledGATv2Projection, ScaledEmbeddedGraphNetwork)
        assert issubclass(ScaledSimpleGATv2Projection, ScaledEmbeddedGraphNetwork)

    def test_gatv2_projection_has_no_public_residual_param(self) -> None:
        sig = inspect.signature(GATv2Projection.__init__)
        assert "residual" not in sig.parameters

    def test_gatv2_projection_uses_residual_message(self) -> None:
        module = GATv2Projection(in_channels=3, out_channels=2, hidden_size=8, num_layers=1)
        message_module = cast(Any, module)._message_module
        assert isinstance(message_module, GraphMessage)
        assert message_module.layers[0].res is not None

    def test_simple_gatv2_projection_uses_plain_message(self) -> None:
        module = SimpleGATv2Projection(in_channels=3, out_channels=2, hidden_size=8, num_layers=1)
        message_module = cast(Any, module)._message_module
        assert isinstance(message_module, SimpleGraphMessage)
        assert message_module.layers[0].res is None

    def test_scaled_simple_gatv2_projection_uses_plain_message(self) -> None:
        module = ScaledSimpleGATv2Projection(
            in_channels=3, out_channels=2, hidden_size=8, num_layers=1
        )
        message_module = cast(Any, module)._message_module
        assert isinstance(message_module, SimpleGraphMessage)
        assert message_module.layers[0].res is None

    def test_forward_shapes(
        self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr_1d: torch.Tensor
    ) -> None:
        residual = GATv2Projection(in_channels=3, out_channels=2, hidden_size=8, num_layers=1)
        plain = SimpleGATv2Projection(in_channels=3, out_channels=2, hidden_size=8, num_layers=1)
        scaled = ScaledGATv2Projection(in_channels=3, out_channels=2, hidden_size=8, num_layers=1)
        scaled_plain = ScaledSimpleGATv2Projection(
            in_channels=3, out_channels=2, hidden_size=8, num_layers=1
        )

        assert residual(node_features, edge_index, edge_attr_1d).shape == (3, 2)
        assert plain(node_features, edge_index, edge_attr_1d).shape == (3, 2)
        assert scaled(node_features, edge_index, edge_attr_1d).shape == (3, 2)
        assert scaled_plain(node_features, edge_index, edge_attr_1d).shape == (3, 2)
