"""Tests for skip connection encoders and decoders.

Tests SkipEncoder1d and SkipDecoder1d for shape transformations,
architecture properties, and round-trip encode/decode.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dlkit.core.models.nn.encoder.skip import SkipEncoder1d, SkipDecoder1d


@pytest.fixture
def enc_channels() -> list[int]:
    """Channel schedule for encoder: [2, 3, 4]."""
    return [2, 3, 4]


@pytest.fixture
def enc_timesteps() -> list[int]:
    """Timestep schedule for encoder: [16, 8, 4]."""
    return [16, 8, 4]


@pytest.fixture
def encoder(enc_channels: list[int], enc_timesteps: list[int]) -> SkipEncoder1d:
    """Skip encoder with minimal channels and sequence length."""
    return SkipEncoder1d(channels=enc_channels, timesteps=enc_timesteps)


@pytest.fixture
def decoder(enc_channels: list[int], enc_timesteps: list[int]) -> SkipDecoder1d:
    """Skip decoder mirroring encoder structure."""
    return SkipDecoder1d(channels=enc_channels[::-1], timesteps=enc_timesteps[::-1])


@pytest.fixture
def enc_input(batch_size: int) -> torch.Tensor:
    """Input tensor matching encoder input spec: (batch, 2, 16)."""
    return torch.randn(batch_size, 2, 16)


@pytest.fixture
def latent_tensor(batch_size: int) -> torch.Tensor:
    """Latent tensor matching encoder output: (batch, 4, 4)."""
    return torch.randn(batch_size, 4, 4)


class TestSkipEncoder1d:
    """Tests for 1D skip connection encoder."""

    def test_output_shape(self, encoder: SkipEncoder1d, enc_input: torch.Tensor) -> None:
        """Encoder should compress to (batch, 4, 4)."""
        out = encoder(enc_input)
        assert out.shape == (enc_input.shape[0], 4, 4)

    def test_is_nn_module(self, encoder: SkipEncoder1d) -> None:
        """Encoder should be an nn.Module."""
        assert isinstance(encoder, nn.Module)

    def test_has_parameters(self, encoder: SkipEncoder1d) -> None:
        """Encoder should have trainable parameters."""
        params = list(encoder.parameters())
        assert len(params) > 0

    def test_has_layers_attribute(self, encoder: SkipEncoder1d) -> None:
        """Encoder should have layers ModuleList."""
        assert hasattr(encoder, "layers")
        assert isinstance(encoder.layers, nn.ModuleList)
        # Should have num_layers == len(channels) - 1
        assert len(encoder.layers) == 2  # [2→3, 3→4]

    def test_with_batch_norm(self, enc_input: torch.Tensor) -> None:
        """Encoder with batch normalization should work."""
        enc = SkipEncoder1d(
            channels=[2, 3, 4], timesteps=[16, 8, 4], normalize="batch"
        )
        out = enc(enc_input)
        assert out.shape == (enc_input.shape[0], 4, 4)

    def test_with_layer_norm(self, enc_input: torch.Tensor) -> None:
        """Encoder with layer normalization should work."""
        enc = SkipEncoder1d(
            channels=[2, 3, 4], timesteps=[16, 8, 4], normalize="layer"
        )
        out = enc(enc_input)
        assert out.shape == (enc_input.shape[0], 4, 4)

    def test_with_dropout(self, enc_input: torch.Tensor) -> None:
        """Encoder with dropout should work."""
        enc = SkipEncoder1d(
            channels=[2, 3, 4], timesteps=[16, 8, 4], dropout=0.2
        )
        enc.eval()  # Disable dropout during eval
        out = enc(enc_input)
        assert out.shape == (enc_input.shape[0], 4, 4)

    def test_stores_channel_timestep_specs(self, encoder: SkipEncoder1d) -> None:
        """Encoder should store channel and timestep schedules."""
        assert encoder.channels == [2, 3, 4]
        assert encoder.timesteps == [16, 8, 4]

    def test_multiple_layers(self, batch_size: int) -> None:
        """Encoder should support varying number of layers."""
        enc = SkipEncoder1d(
            channels=[2, 4, 8, 16], timesteps=[32, 16, 8, 4]
        )
        x = torch.randn(batch_size, 2, 32)
        out = enc(x)
        assert out.shape == (batch_size, 16, 4)


class TestSkipDecoder1d:
    """Tests for 1D skip connection decoder."""

    def test_output_shape(self, decoder: SkipDecoder1d, latent_tensor: torch.Tensor) -> None:
        """Decoder should expand to (batch, 2, 16)."""
        out = decoder(latent_tensor)
        assert out.shape == (latent_tensor.shape[0], 2, 16)

    def test_decoder_not_encoder_subclass(
        self, decoder: SkipDecoder1d, encoder: SkipEncoder1d
    ) -> None:
        """Decoder should not be a subclass of encoder (independent classes)."""
        assert not isinstance(decoder, SkipEncoder1d)
        assert isinstance(decoder, nn.Module)

    def test_has_regression_layer(self, decoder: SkipDecoder1d) -> None:
        """Decoder should have a final Conv1d regression layer."""
        assert hasattr(decoder, "regression_layer")
        assert isinstance(decoder.regression_layer, nn.Conv1d)

    def test_regression_layer_shape(self, decoder: SkipDecoder1d) -> None:
        """Regression layer should have input channels matching decoder's last channel."""
        # channels = [4, 3, 2] (reversed), so last channel is 2
        assert decoder.regression_layer.in_channels == 2
        assert decoder.regression_layer.out_channels == 2

    def test_encoder_decoder_roundtrip_shape(
        self, encoder: SkipEncoder1d, decoder: SkipDecoder1d, enc_input: torch.Tensor
    ) -> None:
        """Encoder → Decoder should preserve original shape."""
        z = encoder(enc_input)
        out = decoder(z)
        assert out.shape == enc_input.shape

    def test_with_batch_norm(self, latent_tensor: torch.Tensor) -> None:
        """Decoder with batch normalization should work."""
        dec = SkipDecoder1d(
            channels=[4, 3, 2], timesteps=[4, 8, 16], normalize="batch"
        )
        out = dec(latent_tensor)
        assert out.shape == (latent_tensor.shape[0], 2, 16)

    def test_with_dropout(self, latent_tensor: torch.Tensor) -> None:
        """Decoder with dropout should work."""
        dec = SkipDecoder1d(
            channels=[4, 3, 2], timesteps=[4, 8, 16], dropout=0.2
        )
        dec.eval()
        out = dec(latent_tensor)
        assert out.shape == (latent_tensor.shape[0], 2, 16)

    def test_has_layers_attribute(self, decoder: SkipDecoder1d) -> None:
        """Decoder should have layers ModuleList."""
        assert hasattr(decoder, "layers")
        assert isinstance(decoder.layers, nn.ModuleList)

    def test_stores_channel_timestep_specs(self, decoder: SkipDecoder1d) -> None:
        """Decoder should store channel and timestep schedules."""
        assert decoder.channels == [4, 3, 2]
        assert decoder.timesteps == [4, 8, 16]


class TestEncoderDecoderIntegration:
    """Integration tests for encoder-decoder pairs."""

    def test_roundtrip_preserves_tensor_properties(
        self, encoder: SkipEncoder1d, decoder: SkipDecoder1d, enc_input: torch.Tensor
    ) -> None:
        """Roundtrip should preserve batch size."""
        z = encoder(enc_input)
        out = decoder(z)
        assert out.shape[0] == enc_input.shape[0]

    def test_encoder_decoder_parameters_are_independent(
        self, encoder: SkipEncoder1d, decoder: SkipDecoder1d
    ) -> None:
        """Encoder and decoder should have different parameters."""
        enc_params = set(id(p) for p in encoder.parameters())
        dec_params = set(id(p) for p in decoder.parameters())
        assert enc_params.isdisjoint(dec_params)

    def test_loss_can_be_computed_on_roundtrip(
        self, encoder: SkipEncoder1d, decoder: SkipDecoder1d, enc_input: torch.Tensor
    ) -> None:
        """Should be able to compute MSE loss on roundtrip."""
        z = encoder(enc_input)
        out = decoder(z)
        loss = torch.nn.functional.mse_loss(out, enc_input)
        assert loss.requires_grad
        assert loss.ndim == 0
