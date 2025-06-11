import torch
from collections.abc import Callable
from pydantic import ConfigDict, validate_call
from torch import nn

from dlkit.datatypes.dataset import Shape
from dlkit.datatypes.networks import NormalizerName
from dlkit.nn.cae.base import CAE
from dlkit.nn.encoder.skip import SkipEncoder1d, SkipDecoder1d
from dlkit.nn.encoder.latent import (
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
        *,
        shape: Shape,
        latent_channels: int,
        latent_size: int,
        latent_width: int = 1,
        num_layers: int = 3,
        kernel_size: int = 3,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.gelu,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
        transpose: bool = False,
    ):
        super().__init__()
        self.shape = shape
        self.latent_channels = latent_channels
        self.latent_width = latent_width
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.normalize = normalize
        self.dropout = dropout
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
        self.decoder = SkipDecoder1d(
            channels[::-1],
            timesteps[::-1],
            kernel_size=kernel_size,
            normalize=normalize,
            activation=activation,
            dropout=dropout,
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.feature_to_latent(x)

    def decode(self, x):
        x = self.latent_to_feature(x)
        x = self.decoder(x)
        return x
