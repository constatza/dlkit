"""Tests for attention and transformer blocks.

Tests SelfAttentionBlock, TransformerEncoderBlock, TransformerDecoderBlock,
and dimension permutation helpers for temporal data.
"""

from __future__ import annotations

import warnings

import torch

from dlkit.domain.nn.attention.basic import SelfAttentionBlock
from dlkit.domain.nn.attention.transformer import (
    TransformerDecoderBlock,
    TransformerEncoderBlock,
    _batch_first,
    _seq_first,
)


class TestSelfAttentionBlock:
    """Tests for self-attention block."""

    def test_output_shape_with_permute(self, temporal_input: torch.Tensor) -> None:
        """Output should preserve input shape when permute=True."""
        sa = SelfAttentionBlock(embed_dim=2, num_heads=1, permute=True)
        out = sa(temporal_input)
        assert out.shape == temporal_input.shape

    def test_output_shape_without_permute(self, batch_size: int) -> None:
        """Output should preserve shape when permute=False (seq, batch, embed)."""
        sa = SelfAttentionBlock(embed_dim=2, num_heads=1, permute=False)
        x = torch.randn(8, batch_size, 2)  # (time, batch, embed) format
        out = sa(x)
        assert out.shape == x.shape

    def test_configurable_dropout(self) -> None:
        """Dropout should be configurable."""
        sa0 = SelfAttentionBlock(embed_dim=2, num_heads=1, dropout=0.0)
        sa5 = SelfAttentionBlock(embed_dim=2, num_heads=1, dropout=0.5)
        assert sa0.multihead_attn.dropout == 0.0
        assert sa5.multihead_attn.dropout == 0.5

    def test_num_heads_parameter(self) -> None:
        """Should support different numbers of heads."""
        sa1 = SelfAttentionBlock(embed_dim=4, num_heads=1)
        sa2 = SelfAttentionBlock(embed_dim=4, num_heads=2)
        assert sa1.multihead_attn.num_heads == 1
        assert sa2.multihead_attn.num_heads == 2

    def test_permute_attribute_stored(self) -> None:
        """Permute flag should be stored."""
        sa_perm = SelfAttentionBlock(embed_dim=2, num_heads=1, permute=True)
        sa_no_perm = SelfAttentionBlock(embed_dim=2, num_heads=1, permute=False)
        assert sa_perm.permute is True
        assert sa_no_perm.permute is False

    def test_is_nn_module(self) -> None:
        """SelfAttentionBlock should be an nn.Module."""
        from torch import nn

        sa = SelfAttentionBlock(embed_dim=2, num_heads=1)
        assert isinstance(sa, nn.Module)

    def test_has_parameters(self) -> None:
        """SelfAttentionBlock should have trainable parameters."""
        sa = SelfAttentionBlock(embed_dim=2, num_heads=1)
        params = list(sa.parameters())
        assert len(params) > 0

    def test_self_attention_output_differs_from_input(self, batch_size: int) -> None:
        """Self-attention output should generally differ from input."""
        sa = SelfAttentionBlock(embed_dim=4, num_heads=1, permute=True)
        x = torch.randn(batch_size, 4, 8)
        out = sa(x)
        # Attention should transform the input (not be identity)
        assert not torch.allclose(out, x, atol=1e-5)


class TestTransformerEncoderBlock:
    """Tests for transformer encoder block."""

    def test_output_shape_single_layer(self, temporal_input: torch.Tensor) -> None:
        """Encoder should preserve shape with single layer."""
        te = TransformerEncoderBlock(embed_dim=2, num_heads=1, num_layers=1)
        out = te(temporal_input)
        assert out.shape == temporal_input.shape

    def test_output_shape_multi_layer(self, temporal_input: torch.Tensor) -> None:
        """Encoder should preserve shape with multiple layers."""
        te = TransformerEncoderBlock(embed_dim=2, num_heads=1, num_layers=3)
        out = te(temporal_input)
        assert out.shape == temporal_input.shape

    def test_is_nn_module(self) -> None:
        """TransformerEncoderBlock should be an nn.Module."""
        from torch import nn

        te = TransformerEncoderBlock(embed_dim=2, num_heads=1)
        assert isinstance(te, nn.Module)

    def test_has_parameters(self) -> None:
        """TransformerEncoderBlock should have trainable parameters."""
        te = TransformerEncoderBlock(embed_dim=2, num_heads=1)
        params = list(te.parameters())
        assert len(params) > 0

    def test_configurable_num_heads(self) -> None:
        """Should support different numbers of heads."""
        te1 = TransformerEncoderBlock(embed_dim=4, num_heads=1)
        te2 = TransformerEncoderBlock(embed_dim=4, num_heads=2)
        assert te1.transformer_layer.self_attn.num_heads == 1
        assert te2.transformer_layer.self_attn.num_heads == 2
        assert te1.transformer_layer.self_attn.batch_first is True
        assert te2.transformer_layer.self_attn.batch_first is True

    def test_configurable_num_layers(self) -> None:
        """Should support different numbers of layers."""
        te1 = TransformerEncoderBlock(embed_dim=2, num_heads=1, num_layers=1)
        te3 = TransformerEncoderBlock(embed_dim=2, num_heads=1, num_layers=3)
        assert te1.transformer_encoder.num_layers == 1
        assert te3.transformer_encoder.num_layers == 3

    def test_output_differs_from_input(self, temporal_input: torch.Tensor) -> None:
        """Encoder output should differ from input (transformation occurs)."""
        te = TransformerEncoderBlock(embed_dim=2, num_heads=1, num_layers=1)
        out = te(temporal_input)
        # Transformer should transform the input (not be identity)
        assert not torch.allclose(out, temporal_input, atol=1e-5)

    def test_odd_num_heads_avoids_nested_tensor_warning(self, temporal_input: torch.Tensor) -> None:
        """Odd-head configurations should disable the unused fast path without warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            te = TransformerEncoderBlock(embed_dim=2, num_heads=1, num_layers=1)
            te(temporal_input)

        assert all("enable_nested_tensor is True" not in str(warning.message) for warning in caught)


class TestTransformerDecoderBlock:
    """Tests for transformer decoder block."""

    def test_output_shape_self_attention(self, temporal_input: torch.Tensor) -> None:
        """Decoder should preserve shape with self-attention."""
        td = TransformerDecoderBlock(embed_dim=2, num_heads=1)
        out = td(temporal_input)
        assert out.shape == temporal_input.shape

    def test_output_shape_with_memory(self, temporal_input: torch.Tensor) -> None:
        """Decoder should work with memory (encoder output)."""
        td = TransformerDecoderBlock(embed_dim=2, num_heads=1)
        memory = torch.randn_like(temporal_input)
        out = td(temporal_input, memory=memory)
        assert out.shape == temporal_input.shape

    def test_memory_none_uses_self_attention(self, temporal_input: torch.Tensor) -> None:
        """When memory is None, should use input as memory (self-attention)."""
        td = TransformerDecoderBlock(embed_dim=2, num_heads=1)
        out_explicit_self = td(temporal_input, memory=temporal_input)
        out_implicit_self = td(temporal_input, memory=None)
        # Both should produce same shape at least
        assert out_explicit_self.shape == out_implicit_self.shape

    def test_is_nn_module(self) -> None:
        """TransformerDecoderBlock should be an nn.Module."""
        from torch import nn

        td = TransformerDecoderBlock(embed_dim=2, num_heads=1)
        assert isinstance(td, nn.Module)

    def test_has_parameters(self) -> None:
        """TransformerDecoderBlock should have trainable parameters."""
        td = TransformerDecoderBlock(embed_dim=2, num_heads=1)
        params = list(td.parameters())
        assert len(params) > 0

    def test_configurable_num_heads(self) -> None:
        """Should support different numbers of heads."""
        td1 = TransformerDecoderBlock(embed_dim=4, num_heads=1)
        td2 = TransformerDecoderBlock(embed_dim=4, num_heads=2)
        assert td1.transformer_layer.self_attn.num_heads == 1
        assert td2.transformer_layer.self_attn.num_heads == 2
        assert td1.transformer_layer.self_attn.batch_first is True
        assert td2.transformer_layer.self_attn.batch_first is True

    def test_configurable_num_layers(self) -> None:
        """Should support different numbers of layers."""
        td1 = TransformerDecoderBlock(embed_dim=2, num_heads=1, num_layers=1)
        td3 = TransformerDecoderBlock(embed_dim=2, num_heads=1, num_layers=3)
        assert td1.transformer_decoder.num_layers == 1
        assert td3.transformer_decoder.num_layers == 3

    def test_different_memory_shapes_handled(self, batch_size: int) -> None:
        """Decoder should handle different memory shapes (seq_len can differ)."""
        td = TransformerDecoderBlock(embed_dim=4, num_heads=1)
        query = torch.randn(batch_size, 4, 8)  # (batch, embed, time)
        memory = torch.randn(batch_size, 4, 12)  # Different time length
        out = td(query, memory=memory)
        assert out.shape == query.shape


class TestPermuteHelpers:
    """Tests for dimension permutation helper functions."""

    def test_seq_first_shape(self, batch_size: int) -> None:
        """_seq_first should convert (batch, embed, time) → (time, batch, embed)."""
        x = torch.randn(batch_size, 2, 8)
        out = _seq_first(x)
        assert out.shape == (8, batch_size, 2)

    def test_batch_first_shape(self, batch_size: int) -> None:
        """_batch_first should convert (time, batch, embed) → (batch, embed, time)."""
        x = torch.randn(8, batch_size, 2)
        out = _batch_first(x)
        assert out.shape == (batch_size, 2, 8)

    def test_roundtrip_seq_batch_first(self, batch_size: int) -> None:
        """seq_first → batch_first should recover original shape."""
        x = torch.randn(batch_size, 2, 8)
        assert torch.equal(_batch_first(_seq_first(x)), x)

    def test_roundtrip_batch_seq_first(self, batch_size: int) -> None:
        """batch_first → seq_first should recover original shape."""
        x = torch.randn(8, batch_size, 2)
        assert torch.equal(_seq_first(_batch_first(x)), x)

    def test_preserves_values(self, batch_size: int) -> None:
        """Permutation should preserve all values (just reorder)."""
        x = torch.arange(batch_size * 2 * 8, dtype=torch.float).reshape(batch_size, 2, 8)
        out = _seq_first(x)
        # Should have same number of elements
        assert x.numel() == out.numel()
        # All values should be present
        assert torch.allclose(x.reshape(-1).sort()[0], out.reshape(-1).sort()[0])

    def test_seq_first_permutes_last_to_first(self, batch_size: int) -> None:
        """_seq_first should move time (dim 2) to first position."""
        x = torch.randn(batch_size, 2, 8)
        out = _seq_first(x)
        # Original dims: (batch=B, embed=E, time=T) -> (B, E, T)
        # New dims: (time=T, batch=B, embed=E) -> (T, B, E)
        assert out.shape[0] == x.shape[2]
        assert out.shape[1] == x.shape[0]
        assert out.shape[2] == x.shape[1]
