import torch
from collections.abc import Callable
from pydantic import ConfigDict, validate_call
from typing import Literal
from torch import nn

from dlkit.datatypes.dataset import Shape
from dlkit.networks.cae.base import CAE
from dlkit.networks.encoder.skip import SkipEncoder1d
from dlkit.networks.encoder.latent import (
    VectorToTensorBlock,
    TensorToVectorBlock,
)


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
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.gelu,
        normalize: Literal["layer", "batch"] | None = None,
        dropout: float = 0.0,
        transpose: bool = False,
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

        self.encoder = SkipEncoder1d(
            channels,
            timesteps,
            kernel_size=kernel_size,
            normalize=normalize,
            activation=activation,
            dropout=dropout,
        )
        if transpose:
            reduce_dim = timesteps[-1]
        else:
            reduce_dim = channels[-1]
        self.feature_to_latent = TensorToVectorBlock(reduce_dim, latent_size, transpose=transpose)

        # Instantiate latent decoder and feature decoder
        self.latent_to_feature = VectorToTensorBlock(latent_size, (channels[-1], timesteps[-1]))
        self.decoder = SkipEncoder1d(
            channels[::-1],
            timesteps[::-1],
            kernel_size=kernel_size,
            normalize=normalize,
            activation=activation,
            dropout=dropout,
        )
        self.smoothing_layer = nn.Sequential(
            nn.Conv1d(channels[0], channels[0], kernel_size=kernel_size, padding="same"),
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.feature_to_latent(x)

    def decode(self, x):
        x = self.latent_to_feature(x)
        x = self.decoder(x)
        return self.smoothing_layer(x)
