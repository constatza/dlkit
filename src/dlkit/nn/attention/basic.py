"""Self-attention block for temporal data."""

from torch import nn


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 1, permute: bool = True):
        """Self-attention block for temporal data.

        Parameters:
        - channels (int): Number of channels in the input.
        - timesteps (int): Number of timesteps in the input.
        """
        super().__init__()
        self.permute = permute
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=0.1
        )

    def forward(self, x):
        if self.permute:
            x = x.permute(2, 0, 1)
        x, _ = self.multihead_attn(x, x, x)
        if self.permute:
            x = x.permute(1, 2, 0)
        return x
