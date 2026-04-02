from collections.abc import Callable

import torch
from torch import nn

from dlkit.domain.nn.cae.base import CAE
from dlkit.domain.nn.encoder.latent import (
    TensorToVectorBlock,
    VectorToTensorBlock,
)
from dlkit.domain.nn.encoder.skip import SkipDecoder1d, SkipEncoder1d
from dlkit.domain.nn.types import NormalizerName
from dlkit.domain.nn.utils import build_channel_schedule


class SkipCAE1d(CAE):
    """1D Skip Connection Convolutional Autoencoder.

    Args:
        in_channels: Number of input channels.
        in_length: Length of input sequence.
        latent_channels: Number of latent channels.
        latent_size: Size of latent vector.
        latent_width: Width of latent feature (default: 1).
        num_layers: Number of encoder/decoder layers (default: 3).
        kernel_size: Convolution kernel size (default: 3).
        activation: Activation function (default: gelu).
        normalize: Normalization type (default: None).
        dropout: Dropout probability (default: 0.0).
        transpose: Whether to transpose dimensions in latent encoding (default: False).
        dilation: Dilation for convolutions (default: 1).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        in_length: int,
        latent_channels: int,
        latent_size: int,
        latent_width: int = 1,
        num_layers: int = 3,
        kernel_size: int = 3,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.gelu,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
        transpose: bool = False,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        channels = build_channel_schedule(in_channels, latent_channels, num_layers + 1)
        timesteps = build_channel_schedule(in_length, latent_width, num_layers + 1)

        self.encoder = SkipEncoder1d(
            channels,
            timesteps,
            kernel_size=kernel_size,
            normalize=normalize,
            activation=activation,
            dropout=dropout,
            dilation=dilation,
        )

        reduce_dim = timesteps[-1] if transpose else channels[-1]
        self.feature_to_latent = TensorToVectorBlock(reduce_dim, latent_size, transpose=transpose)
        self.latent_to_feature = VectorToTensorBlock(latent_size, (channels[-1], timesteps[-1]))
        self.decoder = SkipDecoder1d(
            channels[::-1],
            timesteps[::-1],
            kernel_size=kernel_size,
            normalize=normalize,
            activation=activation,
            dropout=dropout,
            dilation=dilation,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space.

        Args:
            x: Input tensor of shape (batch, in_channels, in_length).

        Returns:
            Latent representation tensor.
        """
        x = self.encoder(x)
        return self.feature_to_latent(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output space.

        Args:
            x: Latent tensor.

        Returns:
            Decoded output tensor.
        """
        x = self.latent_to_feature(x)
        return self.decoder(x)
