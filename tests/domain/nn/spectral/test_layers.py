"""Tests for SpectralConv1d and FourierLayer."""

from __future__ import annotations

import torch

from dlkit.domain.nn.spectral import ISpectralLayer, SpectralConv1d
from dlkit.domain.nn.spectral.layers import FourierLayer


class TestSpectralConv1d:
    def test_output_shape_matches_input(
        self,
        conv_input: torch.Tensor,
        n_channels: int,
        n_modes: int,
    ) -> None:
        layer = SpectralConv1d(in_channels=n_channels, out_channels=n_channels, n_modes=n_modes)
        out = layer(conv_input)
        assert out.shape == conv_input.shape

    def test_output_shape_with_different_out_channels(
        self,
        conv_input: torch.Tensor,
        n_channels: int,
        spatial_length: int,
        n_modes: int,
    ) -> None:
        out_ch = n_channels * 2
        layer = SpectralConv1d(in_channels=n_channels, out_channels=out_ch, n_modes=n_modes)
        out = layer(conv_input)
        assert out.shape == (conv_input.shape[0], out_ch, spatial_length)

    def test_output_dtype_matches_input(
        self, conv_input: torch.Tensor, n_channels: int, n_modes: int
    ) -> None:
        layer = SpectralConv1d(in_channels=n_channels, out_channels=n_channels, n_modes=n_modes)
        assert layer(conv_input).dtype == conv_input.dtype

    def test_n_modes_property(self, n_channels: int, n_modes: int) -> None:
        layer = SpectralConv1d(in_channels=n_channels, out_channels=n_channels, n_modes=n_modes)
        assert layer.n_modes == n_modes

    def test_satisfies_spectral_layer_protocol(self, n_channels: int, n_modes: int) -> None:
        layer = SpectralConv1d(in_channels=n_channels, out_channels=n_channels, n_modes=n_modes)
        assert isinstance(layer, ISpectralLayer)

    def test_n_modes_capped_at_spectrum_length(self, batch_size: int, n_channels: int) -> None:
        """Layers with n_modes > spectrum length should not raise."""
        short_input = torch.randn(batch_size, n_channels, 4)
        layer = SpectralConv1d(in_channels=n_channels, out_channels=n_channels, n_modes=100)
        out = layer(short_input)
        assert out.shape == short_input.shape

    def test_is_differentiable(
        self, conv_input: torch.Tensor, n_channels: int, n_modes: int
    ) -> None:
        layer = SpectralConv1d(in_channels=n_channels, out_channels=n_channels, n_modes=n_modes)
        x = conv_input.requires_grad_(True)
        loss = layer(x).sum()
        loss.backward()
        assert x.grad is not None


class TestFourierLayer:
    def test_output_shape_preserved(
        self, conv_input: torch.Tensor, n_channels: int, n_modes: int
    ) -> None:
        layer = FourierLayer(channels=n_channels, n_modes=n_modes)
        assert layer(conv_input).shape == conv_input.shape

    def test_n_modes_property(self, n_channels: int, n_modes: int) -> None:
        layer = FourierLayer(channels=n_channels, n_modes=n_modes)
        assert layer.n_modes == n_modes

    def test_satisfies_spectral_layer_protocol(self, n_channels: int, n_modes: int) -> None:
        layer = FourierLayer(channels=n_channels, n_modes=n_modes)
        assert isinstance(layer, ISpectralLayer)

    def test_custom_activation(
        self, conv_input: torch.Tensor, n_channels: int, n_modes: int
    ) -> None:
        layer = FourierLayer(channels=n_channels, n_modes=n_modes, activation=torch.relu)
        out = layer(conv_input)
        assert out.shape == conv_input.shape
        assert (out >= 0).all()
