"""Tests for TransformerEncoderBlock and TransformerDecoderBlock."""

from __future__ import annotations

import warnings

import torch

from dlkit.domain.nn.attention.transformer import TransformerDecoderBlock, TransformerEncoderBlock


def test_encoder_uses_pre_ln() -> None:
    """Pre-LN should be enabled for encoder."""
    block = TransformerEncoderBlock(embed_dim=8, num_heads=2, num_layers=1)
    assert block.transformer_layer.norm_first is True


def test_decoder_uses_pre_ln() -> None:
    """Pre-LN should be enabled for decoder."""
    block = TransformerDecoderBlock(embed_dim=8, num_heads=2, num_layers=1)
    assert block.transformer_layer.norm_first is True


def test_encoder_output_shape(transformer_input: torch.Tensor) -> None:
    """Encoder should preserve input shape."""
    block = TransformerEncoderBlock(embed_dim=8, num_heads=2, num_layers=2)
    assert block(transformer_input).shape == transformer_input.shape


def test_encoder_even_heads_avoids_nested_tensor_warning(transformer_input: torch.Tensor) -> None:
    """Even-head pre-LN encoders should disable the unused nested-tensor fast path."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        block = TransformerEncoderBlock(embed_dim=8, num_heads=2, num_layers=2)
        _ = block(transformer_input)

    assert all("enable_nested_tensor is True" not in str(warning.message) for warning in caught)


def test_decoder_output_shape(transformer_input: torch.Tensor) -> None:
    """Decoder should preserve input shape."""
    block = TransformerDecoderBlock(embed_dim=8, num_heads=2, num_layers=2)
    assert block(transformer_input).shape == transformer_input.shape


def test_decoder_is_causal_by_default(transformer_input: torch.Tensor) -> None:
    """Regression: causal=True (default) must block leakage from future timesteps.

    Perturbing positions after t must not change the output at t, when x is
    used as both target and memory (self-attention/autoregressive mode).
    """
    torch.manual_seed(0)
    block = TransformerDecoderBlock(embed_dim=8, num_heads=2, num_layers=2)
    block.eval()

    perturbed = transformer_input.clone()
    perturbed[:, :, 8:] += 100.0

    with torch.no_grad():
        out_base = block(transformer_input)
        out_perturbed = block(perturbed)

    torch.testing.assert_close(out_base[:, :, :8], out_perturbed[:, :, :8])


def test_encoder_exposes_dropout_and_dim_feedforward() -> None:
    """dropout/dim_feedforward must reach the underlying TransformerEncoderLayer."""
    block = TransformerEncoderBlock(embed_dim=8, num_heads=2, dropout=0.3, dim_feedforward=32)
    assert block.transformer_layer.linear1.out_features == 32
    assert block.transformer_layer.dropout.p == 0.3


def test_decoder_exposes_dropout_and_dim_feedforward() -> None:
    """dropout/dim_feedforward must reach the underlying TransformerDecoderLayer."""
    block = TransformerDecoderBlock(embed_dim=8, num_heads=2, dropout=0.3, dim_feedforward=32)
    assert block.transformer_layer.linear1.out_features == 32
    assert block.transformer_layer.dropout.p == 0.3


def test_decoder_causal_false_allows_future_leakage(transformer_input: torch.Tensor) -> None:
    """causal=False must NOT block future timesteps (bidirectional self-attention)."""
    torch.manual_seed(0)
    block = TransformerDecoderBlock(embed_dim=8, num_heads=2, num_layers=2, causal=False)
    block.eval()

    perturbed = transformer_input.clone()
    perturbed[:, :, 8:] += 100.0

    with torch.no_grad():
        out_base = block(transformer_input)
        out_perturbed = block(perturbed)

    assert not torch.allclose(out_base[:, :, :8], out_perturbed[:, :, :8])
