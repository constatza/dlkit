from collections.abc import Sequence
import torch
from pydantic import validate_call, ConfigDict
from torch import nn
from torch.nn import Sequential

from dlkit.networks.blocks.latent import (
    TensorToVectorBlock,
    VectorToTensorBlock,
)
from dlkit.networks.caes.base import CAE

from dlkit.networks.blocks.residual import SkipConnection
from dlkit.networks.blocks.convolutional import ConvolutionBlock1d
from dlkit.settings.general_settings import ModelSettings
from dlkit.utils.math_utils import linear_interpolation_int
from dlkit.metrics.temporal import mase


class SkipCAE1d(CAE):

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        *args,
        activation: nn.Module = nn.GELU(),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(
            ignore=["activation"],
        )

        self.activation = activation
        latent_channels = self.settings.latent_channels
        latent_width = self.settings.latent_width
        latent_size = self.settings.latent_size
        num_layers = self.settings.num_layers
        kernel_size = self.settings.kernel_size
        shape = self.settings.shape
        if len(shape.features) != 2:
            raise ValueError("Shape must be 2D")

        self.example_input_array = torch.randn(1, *shape.features)

        initial_channels = shape.features[-2]
        initial_width = shape.features[-1]

        channels = linear_interpolation_int(
            [initial_channels, latent_channels], num_layers + 1
        )
        channels[1:] = channels[1:] + (channels[1:] % 2)
        channels = channels.tolist()

        timesteps = linear_interpolation_int(
            [initial_width, latent_width], num_layers + 1
        ).tolist()

        # Instantiate feature extractor and latent encoder

        self.encoder = SkipEncoder(
            channels=channels.copy(),
            timesteps=timesteps,
            latent_dim=latent_size,
            kernel_size=kernel_size,
        )

        # Instantiate latent decoder and feature decoder
        self.decoder = SkipDecoder(
            channels=channels.copy(),
            timesteps=timesteps,
            latent_dim=latent_size,
            kernel_size=kernel_size,
        )
        self.smoothing_layer = nn.Sequential(
            nn.SELU(),
            nn.Conv1d(
                channels[0], channels[0], kernel_size=kernel_size, padding="same"
            ),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        x = self.decoder(x)
        return self.smoothing_layer(x)

    @staticmethod
    def training_loss_func(x_hat, x):
        return mase(x_hat, x)

    @staticmethod
    def test_loss_func(x_hat, x):
        return mase(x_hat, x)


class SkipEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channels: Sequence[int],
        kernel_size: int = 3,
        timesteps: Sequence[int] = None,
    ):
        """
        Complete encoder that compresses the input into a latent vector.

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
                    ConvolutionBlock1d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        in_timesteps=timesteps[i],
                        kernel_size=kernel_size,
                        padding="same",
                        dilation=2**i,
                    ),
                )
            )
            layers.append(nn.AdaptiveMaxPool1d(timesteps[i + 1]))

        self.feature_extractor = Sequential(*layers)

        self.feature_to_latent = TensorToVectorBlock(
            channels[-1], timesteps[-1], latent_dim
        )

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
        """
        Complete decoder that reconstructs the input from a latent vector.

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

        self.latent_to_feature = VectorToTensorBlock(
            latent_dim, (channels[0], timesteps[0])
        )

        layers = []
        for i in range(num_layers):
            layers.append(
                SkipConnection(
                    ConvolutionBlock1d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        in_timesteps=timesteps[i],
                        kernel_size=kernel_size,
                        padding="same",
                        dilation=2**i,
                    ),
                )
            )
            layers.append(nn.Upsample(timesteps[i + 1]))

        self.feature_decoder = Sequential(*layers)

    def forward(self, x):
        x = self.latent_to_feature(x)
        x = self.feature_decoder(x)
        return x
