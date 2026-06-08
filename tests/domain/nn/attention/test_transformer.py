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
