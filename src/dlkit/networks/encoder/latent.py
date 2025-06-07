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
        transpose: bool = False,
    ):
        """Converts the feature map into a latent vector.

        Parameters:
        - input_shape (tuple): Shape of the feature map (channels, timesteps).
        - latent_dim (int): Dimension of the latent vector.
        """
        super().__init__()
        self.activation = F.gelu
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dense_block = DenseBlock(channels_in, latent_dim, activation=lambda x: x)
        self.transpose = transpose

    def forward(self, x):
        if self.transpose:
            x = x.transpose(1, 2)
        x = self.pooling(x)
        x = x.flatten(1)
        x = self.dense_block(x)
        return x
