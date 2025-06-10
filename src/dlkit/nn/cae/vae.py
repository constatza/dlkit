from collections.abc import Callable

import torch
from torch import nn
from .skip1d import SkipCAE1d
from pydantic import ConfigDict, validate_call
from dlkit.datatypes.dataset import Shape
from dlkit.datatypes.networks import NormalizerName


class VAE(SkipCAE1d):
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
        super().__init__(
            shape=shape,
            latent_channels=latent_channels,
            latent_size=2 * latent_size,
            latent_width=latent_width,
            num_layers=num_layers,
            kernel_size=kernel_size,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
            transpose=transpose,
        )
        self.save_hyperparameters(
            ignore=["activation"],
        )

        self.mu_layer = nn.Linear(2 * self.latent_size, self.latent_size)
        self.logvar_layer = nn.Linear(2 * self.latent_size, self.latent_size)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
