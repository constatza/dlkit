"""Tests for DeepONet and its convenience variants."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn

from dlkit.common.shapes import ShapeContext
from dlkit.domain.nn.ffnn.residual import FFNN
from dlkit.domain.nn.operators import (
    DeepONet,
    EmbeddedDeepONet,
    FFNNDeepONet,
    IOperatorNetwork,
    IQueryOperator,
    VarWidthDeepONet,
)


class TestDeepONetShapes:
    def test_scalar_output_canonical_query_shape(
        self,
        deeponet_branch: torch.Tensor,
        deeponet_trunk: torch.Tensor,
        batch_size: int,
        n_queries: int,
    ) -> None:
        model = VarWidthDeepONet(
            branch_in_features=20,
            out_features=1,
            query_dim=1,
            trunk_width=32,
            branch_layers=[32, 24, 32],
            trunk_layers=[16, 32],
        )
        out = model(deeponet_branch, deeponet_trunk)
        assert out.shape == (batch_size, n_queries, 1)

    def test_scalar_output_two_dimensional_coords(
        self,
        deeponet_branch: torch.Tensor,
        batch_size: int,
        n_queries: int,
    ) -> None:
        model = VarWidthDeepONet(
            branch_in_features=20,
            out_features=1,
            query_dim=2,
            trunk_width=32,
            branch_layers=[32, 24, 32],
            trunk_layers=[16, 32],
        )
        y = torch.randn(batch_size, n_queries, 2)
        assert model(deeponet_branch, y).shape == (batch_size, n_queries, 1)

    def test_multi_output_query_shape(
        self,
        deeponet_branch: torch.Tensor,
        batch_size: int,
        n_queries: int,
    ) -> None:
        out_features = 3
        model = VarWidthDeepONet(
            branch_in_features=20,
            out_features=out_features,
            query_dim=2,
            trunk_width=32,
            branch_layers=[32, 48],
            trunk_layers=[24, 24, 32],
        )
        y = torch.randn(batch_size, n_queries, 2)
        assert model(deeponet_branch, y).shape == (batch_size, n_queries, out_features)

    def test_parent_requires_canonical_query_shape(self, deeponet_branch: torch.Tensor) -> None:
        model = VarWidthDeepONet(
            branch_in_features=20,
            out_features=1,
            query_dim=1,
            trunk_width=16,
            branch_layers=[16, 24],
            trunk_layers=[16, 16],
        )
        y = torch.randn(10, 1)
        with pytest.raises(ValueError, match="canonical shape"):
            model(deeponet_branch, y)


class TestDeepONetVariants:
    def test_ffnn_deeponet_uses_regular_vector_and_time_input(
        self,
        deeponet_branch: torch.Tensor,
        batch_size: int,
    ) -> None:
        model = FFNNDeepONet(
            branch_in_features=20,
            out_features=5,
            query_dim=1,
            trunk_width=12,
            branch_hidden_size=24,
            branch_num_layers=3,
            trunk_hidden_size=18,
            trunk_num_layers=2,
        )
        y = torch.randn(batch_size, 1, 1)
        assert model(deeponet_branch, y).shape == (batch_size, 1, 5)

    def test_embedded_deeponet_uses_regular_vector_and_coordinate_input(
        self,
        deeponet_branch: torch.Tensor,
        batch_size: int,
    ) -> None:
        model = EmbeddedDeepONet(
            branch_in_features=20,
            out_features=4,
            query_dim=2,
            trunk_width=10,
            branch_hidden_size=32,
            branch_num_layers=2,
            trunk_hidden_size=28,
            trunk_num_layers=3,
        )
        y = torch.randn(batch_size, 1, 2)
        assert model(deeponet_branch, y).shape == (batch_size, 1, 4)

    def test_variants_support_query_batches(
        self,
        deeponet_branch: torch.Tensor,
        batch_size: int,
        n_queries: int,
    ) -> None:
        for model in (
            FFNNDeepONet(
                branch_in_features=20,
                out_features=2,
                query_dim=2,
                trunk_width=8,
                branch_hidden_size=16,
                branch_num_layers=2,
                trunk_hidden_size=12,
                trunk_num_layers=2,
            ),
            EmbeddedDeepONet(
                branch_in_features=20,
                out_features=2,
                query_dim=2,
                trunk_width=8,
                branch_hidden_size=16,
                branch_num_layers=2,
                trunk_hidden_size=12,
                trunk_num_layers=2,
            ),
        ):
            y = torch.randn(batch_size, n_queries, 2)
            assert model(deeponet_branch, y).shape == (batch_size, n_queries, 2)

    def test_variants_are_differentiable(
        self,
        deeponet_branch: torch.Tensor,
        batch_size: int,
        n_queries: int,
    ) -> None:
        u = deeponet_branch.requires_grad_(True)
        y = torch.randn(batch_size, n_queries, 2, requires_grad=True)
        model = FFNNDeepONet(
            branch_in_features=20,
            out_features=2,
            query_dim=2,
            trunk_width=8,
            branch_hidden_size=16,
            branch_num_layers=2,
            trunk_hidden_size=12,
            trunk_num_layers=2,
        )
        model(u, y).sum().backward()
        assert u.grad is not None
        assert y.grad is not None


@pytest.fixture
def non_flat_branch_input_shapes() -> dict[str, tuple[int, ...]]:
    """Branch (non-flat) and query input shapes keyed by entry name."""
    return {"branch": (1, 100), "query": (32, 2)}


@pytest.fixture
def deeponet_output_shapes() -> dict[str, tuple[int, ...]]:
    """Output shape keyed by target entry name."""
    return {"y": (3,)}


class TestDeepONetContractAndProtocols:
    def test_from_entries_flattens_non_flat_branch_inputs(
        self,
        non_flat_branch_input_shapes: dict[str, tuple[int, ...]],
        deeponet_output_shapes: dict[str, tuple[int, ...]],
    ) -> None:
        model = VarWidthDeepONet.from_context(
            ShapeContext(non_flat_branch_input_shapes, deeponet_output_shapes),
            trunk_width=8,
            branch_layers=[16, 16],
            trunk_layers=[12, 12],
        )
        branch_linear = cast(FFNN, model.branch_net).embedding_layer
        trunk_linear = cast(FFNN, model.trunk_net).embedding_layer
        assert branch_linear.in_features == 100
        assert trunk_linear.in_features == 2

    def test_composable_deeponet_supports_any_branch_and_trunk_modules(
        self,
        batch_size: int,
        n_queries: int,
    ) -> None:
        trunk_width, out_features = 16, 1
        latent_dim = trunk_width * out_features
        branch = nn.Sequential(
            nn.Flatten(1), nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, latent_dim)
        )
        trunk = nn.Linear(1, latent_dim)
        u = torch.randn(batch_size, 2, 10)
        y = torch.randn(batch_size, n_queries, 1)
        model = DeepONet(
            branch_net=branch,
            trunk_net=trunk,
            trunk_width=trunk_width,
            out_features=out_features,
        )
        assert model(u, y).shape == (batch_size, n_queries, 1)

    def test_composable_deeponet_raises_helpful_shape_error(
        self,
        batch_size: int,
        n_queries: int,
    ) -> None:
        branch = nn.Linear(20, 15)
        trunk = nn.Linear(1, 16)
        model = DeepONet(
            branch_net=branch,
            trunk_net=trunk,
            trunk_width=16,
            out_features=1,
        )
        u = torch.randn(batch_size, 20)
        y = torch.randn(batch_size, n_queries, 1)

        with pytest.raises(ValueError, match="branch_net must return shape"):
            model(u, y)

    def test_implements_protocols(self) -> None:
        model = FFNNDeepONet(
            branch_in_features=10,
            out_features=1,
            query_dim=1,
            trunk_width=16,
            branch_hidden_size=16,
            branch_num_layers=2,
            trunk_hidden_size=16,
            trunk_num_layers=2,
        )
        assert isinstance(model, IQueryOperator)
        assert isinstance(model, IOperatorNetwork)
        assert isinstance(model, DeepONet)
