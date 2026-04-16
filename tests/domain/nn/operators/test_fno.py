"""Tests for FourierNeuralOperator1d and GridOperatorBase."""

from __future__ import annotations

import torch
from torch import nn

from dlkit.domain.nn.operators import (
    FourierNeuralOperator1d,
    GridOperatorBase,
    IGridOperator,
    IOperatorNetwork,
)
from dlkit.domain.nn.spectral.layers import FourierLayer


class TestFourierNeuralOperator1d:
    def test_output_shape_matches_input_grid(
        self,
        fno_input: torch.Tensor,
        n_channels: int,
        spatial_length: int,
    ) -> None:
        model = FourierNeuralOperator1d(in_channels=n_channels, out_channels=n_channels, n_modes=8)
        out = model(fno_input)
        assert out.shape == (fno_input.shape[0], n_channels, spatial_length)

    def test_different_out_channels(
        self,
        fno_input: torch.Tensor,
        n_channels: int,
        spatial_length: int,
        batch_size: int,
    ) -> None:
        out_ch = 1
        model = FourierNeuralOperator1d(in_channels=n_channels, out_channels=out_ch, n_modes=8)
        assert model(fno_input).shape == (batch_size, out_ch, spatial_length)

    def test_implements_grid_operator_protocol(self, n_channels: int) -> None:
        model = FourierNeuralOperator1d(in_channels=n_channels, out_channels=n_channels, n_modes=8)
        assert isinstance(model, IGridOperator)

    def test_implements_operator_network_protocol(self, n_channels: int) -> None:
        model = FourierNeuralOperator1d(in_channels=n_channels, out_channels=n_channels, n_modes=8)
        assert isinstance(model, IOperatorNetwork)

    def test_out_features_property(self, n_channels: int) -> None:
        out_ch = 3
        model = FourierNeuralOperator1d(in_channels=n_channels, out_channels=out_ch, n_modes=8)
        assert model.out_features == out_ch

    def test_is_nn_module(self, n_channels: int) -> None:
        model = FourierNeuralOperator1d(in_channels=n_channels, out_channels=n_channels, n_modes=8)
        assert isinstance(model, torch.nn.Module)

    def test_is_differentiable(self, fno_input: torch.Tensor, n_channels: int) -> None:
        model = FourierNeuralOperator1d(in_channels=n_channels, out_channels=n_channels, n_modes=8)
        x = fno_input.requires_grad_(True)
        model(x).sum().backward()
        assert x.grad is not None

    def test_discretisation_invariance(self, n_channels: int, batch_size: int) -> None:
        """Model trained on length 32 should forward on length 64 without error."""
        model = FourierNeuralOperator1d(in_channels=n_channels, out_channels=n_channels, n_modes=8)
        x_fine = torch.randn(batch_size, n_channels, 64)
        out = model(x_fine)
        assert out.shape == (batch_size, n_channels, 64)

    def test_is_grid_operator_base_subclass(self, n_channels: int) -> None:
        model = FourierNeuralOperator1d(in_channels=n_channels, out_channels=n_channels, n_modes=8)
        assert isinstance(model, GridOperatorBase)


class TestGridOperatorBase:
    """Tests for the composable GridOperatorBase scaffold."""

    def test_custom_body_injected(
        self,
        fno_input: torch.Tensor,
        n_channels: int,
        spatial_length: int,
        batch_size: int,
    ) -> None:
        """Any nn.Module body can be injected into the lifting+body+projection scaffold."""
        width = 16
        body = nn.Sequential(
            FourierLayer(channels=width, n_modes=8),
            FourierLayer(channels=width, n_modes=8),
        )
        model = GridOperatorBase(
            body=body, in_channels=n_channels, out_channels=n_channels, width=width
        )
        out = model(fno_input)
        assert out.shape == (batch_size, n_channels, spatial_length)

    def test_different_body_types(
        self, fno_input: torch.Tensor, n_channels: int, batch_size: int, spatial_length: int
    ) -> None:
        """A plain Conv1d body also works — the scaffold is body-agnostic."""
        width = 8
        body = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=1),
        )
        model = GridOperatorBase(body=body, in_channels=n_channels, out_channels=1, width=width)
        out = model(fno_input)
        assert out.shape == (batch_size, 1, spatial_length)

    def test_implements_grid_operator_protocol(self, n_channels: int) -> None:
        body = nn.Identity()
        model = GridOperatorBase(
            body=body, in_channels=n_channels, out_channels=n_channels, width=n_channels
        )
        assert isinstance(model, IGridOperator)
        assert isinstance(model, IOperatorNetwork)

    def test_out_features_property(self, n_channels: int) -> None:
        out_ch = 3
        model = GridOperatorBase(
            body=nn.Identity(),
            in_channels=n_channels,
            out_channels=out_ch,
            width=n_channels,
        )
        assert model.out_features == out_ch

    def test_is_differentiable(self, fno_input: torch.Tensor, n_channels: int) -> None:
        width = 8
        body = nn.Sequential(nn.Conv1d(width, width, 1), nn.ReLU())
        model = GridOperatorBase(
            body=body, in_channels=n_channels, out_channels=n_channels, width=width
        )
        x = fno_input.requires_grad_(True)
        model(x).sum().backward()
        assert x.grad is not None
