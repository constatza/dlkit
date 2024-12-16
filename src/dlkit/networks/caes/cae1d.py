import numpy as np
import torch
from torch import nn
from pydantic import validate_call, ConfigDict
from dlkit.networks.caes.blocks import FeatureToLatent, LatentToFeature
from torch import nn

from dlkit.mytypes import TupleLike
from dlkit.networks.caes.base import CAE
from lightning import LightningModule


def create_encoder(
    channels: list[int],
    num_layers: int,
    kernel_size: int,
    timesteps: list[int],
):

    encoder = nn.ModuleList()

    for i in range(num_layers):
        encoder.append(
            nn.Conv1d(
                channels[i],
                channels[i],
                kernel_size,
                stride=1,
                padding="same",
            )
        )
        if timesteps[-1] < timesteps[0]:
            encoder.append(nn.GELU())
            encoder.append(nn.AdaptiveMaxPool1d(timesteps[i + 1]))

        encoder.append(nn.GELU())
        encoder.append(
            nn.Conv1d(
                channels[i],
                channels[i + 1],
                kernel_size=kernel_size,
                stride=1,
                padding="same",
            )
        )

        encoder.append(nn.GELU())

    encoder.append(
        nn.Conv1d(
            channels[-1],
            channels[-1],
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
    )

    encoder = nn.Sequential(*encoder)
    return encoder


def create_decoder(
    channels: list[int],
    num_layers: int,
    kernel_size: int,
    timesteps: list[int],
    input_shape: TupleLike,
):

    dofs = input_shape[-2]
    channels.reverse()
    timesteps.reverse()

    decoder = nn.ModuleList()

    for i in range(num_layers):
        decoder.append(
            nn.Conv1d(channels[i], channels[i], kernel_size, stride=1, padding="same")
        )

        if timesteps[-1] > timesteps[0]:
            decoder.append(nn.GELU())
            decoder.append(nn.AdaptiveMaxPool1d(timesteps[i + 1]))

        decoder.append(nn.GELU())
        decoder.append(
            nn.Conv1d(
                channels[i],
                channels[i + 1],
                kernel_size=kernel_size,
                stride=1,
                padding="same",
            )
        )
        decoder.append(nn.GELU())

    decoder.append(nn.AvgPool1d(kernel_size=7, stride=1, padding=3))
    decoder.append(nn.GELU())
    decoder.append(nn.Conv1d(dofs, dofs, 7, stride=1, padding="same"))

    decoder = nn.Sequential(*decoder)

    return decoder


class BasicCAE(CAE):

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        input_shape: tuple,
        reduced_channels: int = 10,
        reduced_timesteps: int = 5,
        latent_size: int = 10,
        num_layers: int = 4,
        kernel_size: int = 5,
        lr: float = 1e-3,
        activation: nn.Module = nn.GELU(),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(
            "input_shape",
            "reduced_channels",
            "reduced_timesteps",
            "latent_size",
            "num_layers",
            "kernel_size",
            "lr",
        )

        self.activation = activation
        self.input_shape = input_shape

        self.example_input_array = torch.randn(input_shape[-3:])

        channels = (
            np.linspace(input_shape[-2], reduced_channels, num_layers + 1)
            .astype(int)
            .tolist()
        )

        timesteps = (
            np.linspace(input_shape[-1], reduced_timesteps, num_layers + 1)
            .astype(int)
            .tolist()
        )

        # Instantiate feature extractor and latent encoder

        self.encoder = Encoder(
            input_shape=input_shape,
            latent_dim=latent_size,
            channels=channels,
            kernel_size=kernel_size,
            timesteps=timesteps,
            activation=activation,
        )

        # Instantiate latent decoder and feature decoder
        self.decoder = Decoder(
            latent_dim=latent_size,
            channels=channels,
            kernel_size=kernel_size,
            timesteps=timesteps,
            output_shape=input_shape,
            activation=activation,
        )

        self.smoothing_layer = nn.Sequential(
            nn.Conv1d(
                input_shape[1], input_shape[1], kernel_size=kernel_size, padding="same"
            ),
            nn.SELU(),
            nn.Conv1d(
                input_shape[1], input_shape[1], kernel_size=kernel_size, padding="same"
            ),
        )

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        x = self.smoothing_layer(x)
        return x


class FeatureExtractor(LightningModule):
    def __init__(
        self,
        input_shape: tuple,
        channels: list[int],
        timesteps: list[int],
        kernel_size: int = 3,
        activation: nn.Module = nn.GELU(),
    ):
        """
        Progressive encoder that downsamples using residual convolution blocks.

        Parameters:
        - input_shape (tuple): Shape of the input (batch_size, channels, timesteps).
        - channels (List[int]): List of channels for each layer.
        - kernel_size (int): Kernel size for convolutions.
        - timesteps (List[int]): List of timesteps for adaptive pooling at each layer.
        - activation (nn.Module): Activation function for each block.
        """
        super().__init__()
        self.save_hyperparameters()
        self.input_shape = input_shape

        layers = []
        num_layers = len(channels) - 1

        for i in range(num_layers):
            downsample_timesteps = timesteps[i + 1]
            layers.append(
                ResidualConvBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    out_timesteps=downsample_timesteps,
                    kernel_size=kernel_size,
                    activation=activation,
                )
            )

        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        return self.feature_extractor(x)


class Encoder(LightningModule):
    def __init__(
        self,
        input_shape: tuple,
        latent_dim: int,
        channels: list[int],
        kernel_size: int = 3,
        timesteps: list[int] = None,
        activation: nn.Module = nn.GELU(),
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
        self.feature_extractor = FeatureExtractor(
            input_shape=input_shape,
            channels=channels,
            timesteps=timesteps,
            kernel_size=kernel_size,
            activation=activation,
        )

        # Compute the output shape of the feature extractor for initializing FeatureToLatent
        final_channels, final_timesteps = channels[-1], timesteps[-1]
        self.feature_to_latent = FeatureToLatent(
            (final_channels, final_timesteps), latent_dim
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.feature_to_latent(x)
        return x


class ResidualConvBlock(LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_timesteps: int,
        kernel_size: int = 3,
        activation: nn.Module = nn.GELU(),
        is_decoder: bool = False,
        batch_norm: bool = True,
    ):
        """
        A residual transposed convolutional block with upsampling.

        Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - kernel_size (int): Kernel size for transposed convolutions.
        - stride (int): Stride for transposed convolutions to control upsampling.
        - padding (int): Padding for transposed convolutions.
        - upsample_timesteps (Optional[int]): Target timesteps for adaptive upsampling, None if not upsampling.
        - activation (nn.Module): Activation function to use.
        """
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        if is_decoder:
            self.reshape_timesteps = nn.Upsample(
                size=out_timesteps, mode="linear", align_corners=False
            )
        else:
            self.reshape_timesteps = nn.AdaptiveAvgPool1d(out_timesteps)

        # Residual projection to match dimensions
        self.residual_projection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity()

    def forward(self, x):
        x = self.reshape_timesteps(x)
        residual = self.residual_projection(x)
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.hparams.activation(x)
        return x + residual


class FeatureDecoder(LightningModule):
    def __init__(
        self,
        channels: list[int],
        timesteps: list[int],
        kernel_size: int = 3,
        activation: nn.Module = nn.GELU(),
    ):
        """
        Progressive decoder that upsamples using residual transpose blocks.

        Parameters:
        - channels (List[int]): List of channels for each layer, in reverse order from the encoder.
        - kernel_size (int): Kernel size for transposed convolutions.
        - timesteps (List[int]): List of timesteps for adaptive upsampling at each layer.
        - activation (nn.Module): Activation function to use.
        """
        super().__init__()
        layers = []
        num_layers = len(channels) - 1

        for i in range(num_layers):
            out_timesteps = timesteps[i + 1]

            layers.append(
                ResidualConvBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    out_timesteps=out_timesteps,
                    kernel_size=kernel_size,
                    activation=activation,
                    is_decoder=True,
                )
            )

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Decoder(LightningModule):
    def __init__(
        self,
        latent_dim: int,
        channels: list[int],
        timesteps: list[int],
        kernel_size: int = 3,
        activation: nn.Module = nn.GELU(),
        output_shape: tuple = None,
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
        channels = channels[::-1]
        timesteps = timesteps[::-1]
        self.latent_to_feature = LatentToFeature(
            latent_dim, (channels[0], timesteps[0])
        )
        self.feature_decoder = FeatureDecoder(
            channels=channels,
            timesteps=timesteps,
            kernel_size=kernel_size,
            activation=activation,
        )
        self.output_shape = output_shape

    def forward(self, x):
        x = self.latent_to_feature(x)
        x = self.feature_decoder(x)

        # Ensure the output shape matches the input shape
        target_channels, target_timesteps = self.output_shape[-2:]
        x = nn.functional.interpolate(
            x, size=(target_timesteps,), mode="linear", align_corners=False
        )

        return x
