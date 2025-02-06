import torch
from pydantic import validate_call, ConfigDict
from torch import nn
from torch.nn import Sequential

from dlkit.networks.blocks.latent import (
    TensorToVectorBlock,
    VectorToTensorBlock,
)
from dlkit.networks.caes.base import CAE

from dlkit.networks.blocks.residual import ResidualBlock
from dlkit.networks.blocks.convolutional import (
    ConvSameTimesteps,
    UpsampleTimesteps,
    DownsampleTimesteps,
)
from dlkit.utils.math_utils import linear_interpolation_int
import torch.nn.functional as F


class DiffCAE1d(CAE):

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        input_shape: tuple,
        final_channels: int = 10,
        final_timesteps: int = 5,
        latent_size: int = 10,
        num_layers: int = 4,
        kernel_size: int = 5,
        activation: nn.Module = nn.GELU(),
        *args,
        **kwargs,
    ):
        """
        Initialize a `DiffCAE1d` instance.

        Parameters
        ----------
        input_shape : tuple
            Input shape of the data (batch_size, channels, timesteps).
        final_channels : int, optional
            Number of channels in the final latent encoding, by default 10.
        final_timesteps : int, optional
            Number of timesteps in the final latent encoding, by default 5.
        latent_size : int, optional
            Size of the latent vector, by default 10.
        num_layers : int, optional
            Number of layers in the encoder and decoder, by default 4.
        kernel_size : int, optional
            Kernel size for convolutions, by default 5.
        activation : nn.Module, optional
            Activation function for each block, by default nn.GELU().

        Returns
        -------
        None
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(
            ignore=["activation"],
        )

        self.activation = activation
        self.input_shape = input_shape

        self.example_input_array = torch.randn(input_shape)

        initial_channels = input_shape[-2]
        initial_time_steps = input_shape[-1]

        channels = linear_interpolation_int(
            [initial_channels, final_channels], num_layers + 1
        )
        channels[1:] = channels[1:] + (channels[1:] % 2)
        channels = channels.tolist()

        timesteps = linear_interpolation_int(
            [initial_time_steps, final_timesteps], num_layers + 1
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

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class SkipEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channels: list[int],
        kernel_size: int = 3,
        timesteps: list[int] = None,
    ):
        """
        Complete encoder that compresses the input into a latent vector.

        Parameters:
        - input_shape (tuple): Shape of the input (batch_size, channels, timesteps).
        - latent_dim (int): Dimension of the latent vector.
        - channels (List[int]): List of channels for each layer.
        - kernel_size (int): Kernel size for convolutions.
        - timesteps (List[int]): List of timesteps for adaptive pooling at each layer.
        - activation (nn.Module): Activation function for each block.
        """
        super().__init__()

        self.reduce_dx = ResidualBlock(
            ConvSameTimesteps(
                in_channels=channels[0],
                out_channels=(channels[0] + 1) // 2,
                kernel_size=kernel_size,
                # batch_norm=True,
            ),
            aggregator="sum",
        )

        self.reduce_x = ResidualBlock(
            ConvSameTimesteps(
                in_channels=channels[0],
                out_channels=(channels[0] + 1) // 2,
                kernel_size=kernel_size,
                # batch_norm=True,
            ),
            aggregator="sum",
        )
        channels[0] += channels[0] % 2
        layers = []
        for i in range(len(timesteps) - 1):
            layers.append(
                ResidualBlock(
                    ConvSameTimesteps(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_size,
                        dilation=i + 1,
                        groups=2,
                        # batch_norm=True,
                    ),
                )
            )
            layers.append(DownsampleTimesteps(out_timesteps=timesteps[i + 1]))

        self.feature_extractor = Sequential(*layers)

        self.feature_to_latent = TensorToVectorBlock(
            channels[-1], timesteps[-1], latent_dim, batch_norm=True
        )

    def forward(self, x):
        x = torch.cat((self.reduce_x(x), self.reduce_dx(self.delta(x))), dim=1)
        x = self.feature_extractor(x)
        x = self.feature_to_latent(x)
        return x

    @staticmethod
    def delta(x: torch.Tensor) -> torch.Tensor:
        # central differences
        dx = 0.5 * (x[..., 2:] - x[..., :-2])
        dx = torch.nn.functional.pad(dx, (1, 1), mode="constant", value=0)
        return dx


class SkipDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channels: list[int],
        timesteps: list[int],
        kernel_size: int = 3,
    ):
        """
        Complete decoder that reconstructs the input from a latent vector.

        Parameters:
        - latent_dim (int): Dimension of the latent vector input.
        - channels (List[int]): List of channels for each layer, in reverse order from the encoder.
        - kernel_size (int): Kernel size for transposed convolutions.
        - timesteps (List[int]): List of timesteps for adaptive upsampling.
        - activation (nn.Module): Activation function for each block.
        - output_shape (tuple): Target output shape (batch_size, channels, timesteps) to guarantee correct reconstruction.
        """
        super().__init__()
        timesteps = timesteps[::-1]
        channels = channels[::-1]

        self.latent_to_feature = VectorToTensorBlock(
            latent_dim, (channels[0], timesteps[0]), batch_norm=True
        )

        num_layers = len(timesteps) - 1
        layers = []
        for i in range(num_layers):
            layers.append(
                ResidualBlock(
                    ConvSameTimesteps(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_size,
                        dilation=i + 1,
                        # batch_norm=True,
                    ),
                )
            )
            layers.append(UpsampleTimesteps(out_timesteps=timesteps[i + 1]))

        self.feature_decoder = Sequential(*layers)

        self.smoothing_layer = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(channels[-1], channels[-1], kernel_size=7, padding="same"),
        )

    def forward(self, x):
        x = self.latent_to_feature(x)
        x = self.feature_decoder(x)
        x = self.smoothing_layer(x)
        return x
