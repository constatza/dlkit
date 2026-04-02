"""Self-attention block for temporal"""

import torch
from torch import nn


class SelfAttentionBlock(nn.Module):
    """Self-attention block with optional permutation for temporal data.

    This module applies multihead self-attention, optionally permuting the input
    from (batch, channels, time) to (time, batch, channels) format if requested.
    """

    def __init__(
        self, embed_dim: int, num_heads: int = 1, permute: bool = True, dropout: float = 0.1
    ) -> None:
        """Initialize SelfAttentionBlock.

        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads).
            num_heads: Number of attention heads. Defaults to 1.
            permute: Whether to permute input/output for transformer format. Defaults to True.
            dropout: Dropout probability in attention. Defaults to 0.1.
        """
        super().__init__()
        self.permute = permute
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.

        Args:
            x: Input tensor of shape (batch, channels, time) or (time, batch, channels)
                depending on the permute setting.

        Returns:
            Attention output with the same shape as input.
        """
        if self.permute:
            x = x.permute(2, 0, 1)
        x, _ = self.multihead_attn(x, x, x)
        if not self.permute:
            return x
        return x.permute(1, 2, 0)
