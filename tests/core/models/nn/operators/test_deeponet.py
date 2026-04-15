"""Tests for DeepONet (composable) and MLPDeepONet (convenience)."""

from __future__ import annotations

import torch
from torch import nn

from dlkit.domain.nn.operators import DeepONet, IOperatorNetwork, IQueryOperator, MLPDeepONet

# ---------------------------------------------------------------------------
# Fixtures used across both test classes are in conftest.py
# ---------------------------------------------------------------------------


class TestMLPDeepONet:
    """Tests for the MLP convenience constructor (previous DeepONet API)."""

    def test_output_shape(
        self,
        deeponet_branch: torch.Tensor,
        deeponet_trunk: torch.Tensor,
        batch_size: int,
        n_queries: int,
    ) -> None:
        model = MLPDeepONet(
            in_features=20, out_features=1, n_coords=1, trunk_width=32, hidden_size=32
        )
        out = model(deeponet_branch, deeponet_trunk)
        assert out.shape == (batch_size, n_queries, 1)

    def test_output_shape_multi_output(
        self,
        deeponet_branch: torch.Tensor,
        deeponet_trunk: torch.Tensor,
        batch_size: int,
        n_queries: int,
    ) -> None:
        out_features = 3
        model = MLPDeepONet(
            in_features=20,
            out_features=out_features,
            n_coords=1,
            trunk_width=32,
            hidden_size=32,
        )
        assert model(deeponet_branch, deeponet_trunk).shape == (batch_size, n_queries, out_features)

    def test_broadcast_trunk_2d(
        self,
        deeponet_branch: torch.Tensor,
        n_queries: int,
        n_coords: int,
        batch_size: int,
    ) -> None:
        model = MLPDeepONet(
            in_features=20, out_features=1, n_coords=n_coords, trunk_width=32, hidden_size=32
        )
        y_broadcast = torch.randn(n_queries, n_coords)
        assert model(deeponet_branch, y_broadcast).shape == (batch_size, n_queries, 1)

    def test_is_query_operator(self) -> None:
        model = MLPDeepONet(
            in_features=10, out_features=1, n_coords=1, trunk_width=16, hidden_size=16
        )
        assert isinstance(model, IQueryOperator)
        assert isinstance(model, IOperatorNetwork)
        assert isinstance(model, DeepONet)

    def test_out_features_property(self) -> None:
        model = MLPDeepONet(
            in_features=10, out_features=5, n_coords=2, trunk_width=16, hidden_size=16
        )
        assert model.out_features == 5

    def test_is_differentiable(
        self,
        deeponet_branch: torch.Tensor,
        deeponet_trunk: torch.Tensor,
    ) -> None:
        model = MLPDeepONet(
            in_features=20, out_features=1, n_coords=1, trunk_width=32, hidden_size=32
        )
        u = deeponet_branch.requires_grad_(True)
        y = deeponet_trunk.requires_grad_(True)
        model(u, y).sum().backward()
        assert u.grad is not None
        assert y.grad is not None


class TestDeepONetComposable:
    """Tests for the composable DeepONet base with injected branch/trunk."""

    def test_custom_branch_and_trunk(
        self,
        deeponet_branch: torch.Tensor,
        deeponet_trunk: torch.Tensor,
        batch_size: int,
        n_queries: int,
    ) -> None:
        trunk_width, out_features = 16, 1
        latent_dim = trunk_width * out_features
        branch = nn.Linear(20, latent_dim)
        trunk = nn.Linear(1, latent_dim)
        model = DeepONet(
            branch_net=branch, trunk_net=trunk, trunk_width=trunk_width, out_features=out_features
        )
        out = model(deeponet_branch, deeponet_trunk)
        assert out.shape == (batch_size, n_queries, out_features)

    def test_any_nn_module_as_branch(
        self,
        deeponet_branch: torch.Tensor,
        deeponet_trunk: torch.Tensor,
        batch_size: int,
        n_queries: int,
    ) -> None:
        """Branch net can be any nn.Module — here a 2-layer MLP."""
        trunk_width, out_features = 8, 1
        latent_dim = trunk_width * out_features
        branch = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, latent_dim))
        trunk = nn.Linear(1, latent_dim)
        model = DeepONet(
            branch_net=branch, trunk_net=trunk, trunk_width=trunk_width, out_features=out_features
        )
        assert model(deeponet_branch, deeponet_trunk).shape == (batch_size, n_queries, out_features)

    def test_implements_protocols(self) -> None:
        branch = nn.Linear(10, 16)
        trunk = nn.Linear(1, 16)
        model = DeepONet(branch_net=branch, trunk_net=trunk, trunk_width=16, out_features=1)
        assert isinstance(model, IQueryOperator)
        assert isinstance(model, IOperatorNetwork)

    def test_out_features_property(self) -> None:
        branch = nn.Linear(10, 32)
        trunk = nn.Linear(1, 32)
        model = DeepONet(branch_net=branch, trunk_net=trunk, trunk_width=16, out_features=2)
        assert model.out_features == 2

    def test_protocols_are_correctly_satisfied(self) -> None:
        """Runtime isinstance checks: each class satisfies its intended protocol.

        Note: @runtime_checkable Protocol checks attribute *existence*, not
        arity — cross-protocol negative isinstance is not meaningful at runtime.
        """
        from dlkit.domain.nn.operators import FourierNeuralOperator1d, IGridOperator

        branch = nn.Linear(10, 16)
        trunk = nn.Linear(1, 16)
        deep_o = DeepONet(branch_net=branch, trunk_net=trunk, trunk_width=16, out_features=1)
        fno = FourierNeuralOperator1d(in_channels=2, out_channels=2, n_modes=4)

        assert isinstance(deep_o, IQueryOperator)
        assert isinstance(deep_o, IOperatorNetwork)
        assert isinstance(fno, IGridOperator)
        assert isinstance(fno, IOperatorNetwork)
