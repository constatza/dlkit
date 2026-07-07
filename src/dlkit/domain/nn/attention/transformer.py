"""Transformer blocks for temporal"""

import torch
from torch import nn


def _encoder_supports_nested_tensor(layer: nn.TransformerEncoderLayer) -> bool:
    """Return whether the current encoder layer configuration can use nested tensors."""
    return not layer.norm_first


def _seq_first(x: torch.Tensor) -> torch.Tensor:
    """Convert (batch, channels, time) → (time, batch, channels) for transformer input.

    Args:
        x: Input tensor of shape (batch, channels, time).

    Returns:
        Permuted tensor of shape (time, batch, channels).
    """
    return x.permute(2, 0, 1)


def _batch_first(x: torch.Tensor) -> torch.Tensor:
    """Convert (time, batch, channels) → (batch, channels, time) after transformer output.

    Args:
        x: Input tensor of shape (time, batch, channels).

    Returns:
        Permuted tensor of shape (batch, channels, time).
    """
    return x.permute(1, 2, 0)


def switch_channels_with_time(x: torch.Tensor) -> torch.Tensor:
    """Swap the channel and time axes in a batch-first temporal tensor."""

    return x.permute(0, 2, 1)


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block for temporal data.

    Applies multi-layer transformer encoder, automatically handling dimension
    permutation for (batch, channels, time) input format.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        num_layers: int = 1,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
    ) -> None:
        """Initialize TransformerEncoderBlock.

        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads).
            num_heads: Number of attention heads. Defaults to 1.
            num_layers: Number of transformer encoder layers. Defaults to 1.
            dropout: Dropout probability (attention weights and each
                sublayer's output). Defaults to 0.1, matching
                ``nn.TransformerEncoderLayer``'s own default.
            dim_feedforward: Width of the feedforward sublayer. Defaults to
                2048, matching ``nn.TransformerEncoderLayer``'s own default.
        """
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=num_layers,
            enable_nested_tensor=_encoder_supports_nested_tensor(self.transformer_layer),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer encoding.

        Args:
            x: Input tensor of shape (batch, channels, time).

        Returns:
            Encoded tensor of shape (batch, channels, time).
        """
        return switch_channels_with_time(self.transformer_encoder(switch_channels_with_time(x)))


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block for temporal data.

    Applies multi-layer transformer decoder with cross-attention, automatically
    handling dimension permutation for (batch, channels, time) input format.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        num_layers: int = 1,
        causal: bool = True,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
    ) -> None:
        """Initialize TransformerDecoderBlock.

        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads).
            num_heads: Number of attention heads. Defaults to 1.
            num_layers: Number of transformer decoder layers. Defaults to 1.
            causal: Whether to mask future target positions (autoregressive
                decoding). Defaults to ``True``; set ``False`` for bidirectional
                (non-autoregressive) target self-attention.
            dropout: Dropout probability (attention weights and each
                sublayer's output). Defaults to 0.1, matching
                ``nn.TransformerDecoderLayer``'s own default.
            dim_feedforward: Width of the feedforward sublayer. Defaults to
                2048, matching ``nn.TransformerDecoderLayer``'s own default.
        """
        super().__init__()
        self.causal = causal
        self.transformer_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_layer, num_layers=num_layers
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> torch.Tensor:
        """Apply transformer decoding.

        Args:
            x: Input tensor (query) of shape (batch, channels, time).
            memory: Memory tensor (encoder output) of shape (batch, channels, time).
                If None, uses x as memory (self-attention). Defaults to None.

        Returns:
            Decoded tensor of shape (batch, channels, time).
        """
        is_self_attention = memory is None
        mem = x if is_self_attention else memory
        tgt = switch_channels_with_time(x)
        causal_mask = (
            nn.Transformer.generate_square_subsequent_mask(tgt.shape[1], device=tgt.device)
            if self.causal
            else None
        )
        # In self-attention mode (memory=x), cross-attention would otherwise see the
        # full (unmasked) sequence and leak future positions around the tgt_mask.
        memory_mask = causal_mask if (self.causal and is_self_attention) else None
        return switch_channels_with_time(
            self.transformer_decoder(
                tgt,
                switch_channels_with_time(mem),
                tgt_mask=causal_mask,
                memory_mask=memory_mask,
                tgt_is_causal=self.causal,
            )
        )
