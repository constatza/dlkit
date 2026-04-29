"""Transformer blocks for temporal"""

import torch
from torch import nn


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

    def __init__(self, embed_dim: int, num_heads: int = 1, num_layers: int = 1) -> None:
        """Initialize TransformerEncoderBlock.

        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads).
            num_heads: Number of attention heads. Defaults to 1.
            num_layers: Number of transformer encoder layers. Defaults to 1.
        """
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=num_layers,
            enable_nested_tensor=num_heads % 2 == 0,
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

    def __init__(self, embed_dim: int, num_heads: int = 1, num_layers: int = 1) -> None:
        """Initialize TransformerDecoderBlock.

        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads).
            num_heads: Number of attention heads. Defaults to 1.
            num_layers: Number of transformer decoder layers. Defaults to 1.
        """
        super().__init__()
        self.transformer_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
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
        mem = x if memory is None else memory
        return switch_channels_with_time(
            self.transformer_decoder(
                switch_channels_with_time(x),
                switch_channels_with_time(mem),
            )
        )
