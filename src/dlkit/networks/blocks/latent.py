import torch
import torch.nn.functional as F
from torch import nn

from dlkit.networks.blocks.dense import DenseBlock


class VectorToTensorBlock(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        target_shape: tuple,
    ):
        """Converts latent vector into a feature map for the decoder.

        Parameters:
        - latent_dim (int): Dimension of the latent vector.
        - target_shape (tuple): Target shape as (channels, timesteps) for the feature map.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.target_shape = target_shape
        self.dense_block = DenseBlock(
            latent_dim,
            target_shape[0] * target_shape[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        return x.view(x.size(0), *self.target_shape)


class TensorToVectorBlock(nn.Module):
    def __init__(
        self,
        channels_in: int,
        latent_dim: int,
    ):
        """Converts the feature map into a latent vector.

        Parameters:
        - input_shape (tuple): Shape of the feature map (channels, timesteps).
        - latent_dim (int): Dimension of the latent vector.
        """
        super().__init__()
        self.activation = F.gelu
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dense_block = DenseBlock(channels_in, latent_dim)

    def forward(self, x):
        x = self.pooling(x)
        x = x.flatten(1)
        x = self.dense_block(x)
        return x


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
