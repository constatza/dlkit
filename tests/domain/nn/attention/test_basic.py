"""Tests for SelfAttentionBlock."""

from __future__ import annotations

import torch

from dlkit.domain.nn.attention.basic import SelfAttentionBlock


def test_self_attention_output_shape(attention_input: torch.Tensor) -> None:
    """Test that SelfAttentionBlock preserves input shape."""
    block = SelfAttentionBlock(embed_dim=8, num_heads=2)
    assert block(attention_input).shape == (3, 8, 16)


def test_self_attention_has_layer_norm() -> None:
    """Test that SelfAttentionBlock has a LayerNorm component."""
    block = SelfAttentionBlock(embed_dim=8, num_heads=1)
    assert hasattr(block, "norm"), "SelfAttentionBlock must have a LayerNorm"
    assert isinstance(block.norm, torch.nn.LayerNorm)


def test_self_attention_residual(attention_input_no_permute: torch.Tensor) -> None:
    """Test that residual connection + LayerNorm changes output from input."""
    torch.manual_seed(42)
    block = SelfAttentionBlock(embed_dim=8, num_heads=1, permute=False)
    out = block(attention_input_no_permute)
    assert out.shape == attention_input_no_permute.shape
    assert not torch.allclose(out, attention_input_no_permute)
