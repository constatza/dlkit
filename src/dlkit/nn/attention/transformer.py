"""Transformer blocks for temporal data."""

from torch import nn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 1, num_layers: int = 1):
        """Transformer block for temporal data.

        Parameters:
        - embed_dim (int): Embedding dimension.
        - num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 1, num_layers: int = 1):
        """Transformer block for temporal data.

        Parameters:
        - embed_dim (int): Embedding dimension.
        - num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.transformer_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_layer, num_layers=num_layers
        )

    def forward(self, x, memory=None):
        if memory is None:
            memory = x
        x = x.permute(2, 0, 1)
        x = self.transformer_decoder(x, memory.permute(2, 0, 1))
        x = x.permute(1, 2, 0)
        return x
