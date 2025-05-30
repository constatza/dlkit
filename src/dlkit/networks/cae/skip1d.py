from collections.abc import Sequence

import torch
from pydantic import ConfigDict, validate_call
from torch import nn
from torch.nn import Sequential

from dlkit.datatypes.dataset import Shape
from dlkit.networks.blocks.convolutional import ConvolutionBlock1d
from dlkit.networks.blocks.latent import (
    TensorToVectorBlock,
    VectorToTensorBlock,
)
from dlkit.networks.blocks.residual import SkipConnection
from dlkit.networks.cae.base import CAE


class SkipCAE1d(CAE):
    latent_channels: int
    latent_width: int
    latent_size: int
    num_layers: int
    kernel_size: int
    shape: Shape

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        shape: Shape,
        latent_channels: int = 5,
        latent_width: int = 10,
        latent_size: int = 10,
        num_layers: int = 3,
        kernel_size: int = 3,
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["activation"],
        )

        self.activation = activation
        initial_channels = shape.features[0]
        initial_width = shape.features[1]

        channels = torch.linspace(initial_channels, latent_channels, num_layers + 1).int().tolist()
        timesteps = torch.linspace(initial_width, latent_width, num_layers + 1).int().tolist()

        # Instantiate feature extractor and latent encoder

        self.encoder = SkipEncoder(
            channels=channels,
            timesteps=timesteps,
            latent_dim=latent_size,
            kernel_size=kernel_size,
        )

        # Instantiate latent decoder and feature decoder
        self.decoder = SkipDecoder(
            channels=channels,
            timesteps=timesteps,
            latent_dim=latent_size,
            kernel_size=kernel_size,
        )
        self.smoothing_layer = nn.Sequential(
            nn.SELU(),
            nn.Conv1d(channels[0], channels[0], kernel_size=kernel_size, padding="same"),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        x = self.decoder(x)
        return self.smoothing_layer(x)


class SkipEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channels: Sequence[int],
        kernel_size: int = 3,
        timesteps: Sequence[int] | None = None,
    ):
        """Complete encoder that compresses the input into a latent vector.

        Parameters:
        - latent_dim (int): Dimension of the latent vector.
        - channels (List[int]): List of channels for each layer.
        - kernel_size (int): Kernel size for convolutions.
        - timesteps (List[int]): List of timesteps for adaptive pooling at each layer.
        """
        super().__init__()

        layers = []
        for i in range(len(timesteps) - 1):
            layers.append(
                SkipConnection(
                    Sequential(
                        ConvolutionBlock1d(
                            in_channels=channels[i],
                            out_channels=channels[i + 1],
                            in_timesteps=timesteps[i],
                            kernel_size=kernel_size,
                            padding="same",
                            dilation=i + 1,
                        ),
                    ),
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                )
            )
            layers.append(nn.AdaptiveMaxPool1d(timesteps[i + 1]))

        self.feature_extractor = Sequential(*layers)

        self.feature_to_latent = TensorToVectorBlock(channels[-1], latent_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.feature_to_latent(x)
        return x


class SkipDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channels: Sequence[int],
        timesteps: Sequence[int],
        kernel_size: int = 3,
    ):
        """Complete decoder that reconstructs the input from a latent vector.

        Parameters:
        - latent_dim (int): Dimension of the latent vector input.
        - channels (List[int]): List of channels for each layer, in reverse order from the encoder.
        - kernel_size (int): Kernel size for transposed convolutions.
        - timesteps (List[int]): List of timesteps for adaptive upsampling.
        """
        super().__init__()
        num_layers = len(timesteps) - 1
        timesteps = timesteps[::-1]
        channels = channels[::-1]

        self.latent_to_feature = VectorToTensorBlock(latent_dim, (channels[0], timesteps[0]))

        layers = []
        for i in range(num_layers):
            layers.append(
                SkipConnection(
                    Sequential(
                        ConvolutionBlock1d(
                            in_channels=channels[i],
                            out_channels=channels[i + 1],
                            in_timesteps=timesteps[i],
                            kernel_size=kernel_size,
                            padding="same",
                            dilation=i + 1,
                        ),
                    ),
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                )
            )
            layers.append(nn.Upsample(timesteps[i + 1]))

        self.feature_decoder = Sequential(*layers)

    def forward(self, x):
        x = self.latent_to_feature(x)
        x = self.feature_decoder(x)
        return x

    # @staticmethod
    # def training_loss_func(x_hat, x):
    #     return mase(x_hat, x)
    #
    # @staticmethod
    # def test_loss_func(x_hat, x):
    #     return mase(x_hat, x)
