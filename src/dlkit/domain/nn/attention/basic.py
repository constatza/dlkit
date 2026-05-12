"""Self-attention block for temporal"""

import torch
from torch import nn


class SelfAttentionBlock(nn.Module):
    """Self-attention block with LayerNorm and residual connection.

    This module applies multihead self-attention with layer normalization and
    residual connection, optionally permuting the input from (batch, channels, time)
    to (batch, time, channels) format if requested.
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
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention with residual connection and layer normalization.

        Args:
            x: Input tensor of shape (batch, channels, time) if permute=True,
                or (batch, time, embed_dim) if permute=False.

        Returns:
            Attention output with the same shape as input.
        """
        if self.permute:
            x = x.permute(0, 2, 1)  # (B,C,T) → (B,T,C)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        if self.permute:
            x = x.permute(0, 2, 1)  # (B,T,C) → (B,C,T)
        return x
