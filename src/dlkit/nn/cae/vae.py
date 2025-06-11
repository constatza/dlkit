from collections.abc import Callable

import torch
from torch import nn
from pydantic import ConfigDict, validate_call
from dlkit.datatypes.dataset import Shape
from dlkit.datatypes.networks import NormalizerName
from dlkit.nn.cae.base import CAE
from dlkit.nn.encoder.skip import SkipEncoder1d, SkipDecoder1d
from dlkit.nn.encoder.latent import TensorToVectorBlock, VectorToTensorBlock
from torch.distributions.normal import Normal


class VAE1d(CAE):
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
        scale_of_latent: int = 4,
        alpha: float = 1.0,
        beta: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        initial_channels = shape.features[0]
        initial_width = shape.features[1]

        channels = torch.linspace(initial_channels, latent_channels, num_layers + 1).int().tolist()
        timesteps = torch.linspace(initial_width, latent_width, num_layers + 1).int().tolist()
        self.encoder = SkipEncoder1d(
            channels=channels,
            timesteps=timesteps,
            kernel_size=kernel_size,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )
        self.vectorize = TensorToVectorBlock(
            channels_in=channels[-1],
            latent_dim=scale_of_latent * latent_size,
        )
        self.tensorize = VectorToTensorBlock(
            latent_dim=latent_size,
            target_shape=(channels[-1], timesteps[-1]),
        )
        self.decoder = SkipDecoder1d(
            channels=channels[::-1],
            timesteps=timesteps[::-1],
            kernel_size=kernel_size,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )

        self.latent_size = latent_size

        self.mu_layer = nn.Linear(scale_of_latent * self.latent_size, self.latent_size)
        self.logvar_layer = nn.Linear(scale_of_latent * self.latent_size, self.latent_size)

    def encode(self, x):
        x = self.encoder(x)
        x = self.vectorize(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        y, mu, logvar = self.decode(mu, logvar)
        return y, mu, logvar

    def decode(self, mu, logvar):
        z = reparameterize(mu, logvar)
        y = self.tensorize(z)
        y = self.decoder(y)
        return y, mu, logvar

    def predict_step(self, batch, batch_idx):
        x = batch[0]
        x, mu, logvar = self.forward(x)
        return {"predictions": x.detach(), "latent": mu.detach()}

    def loss_function(self, predictions, targets, mu, logvar):
        mse_loss = nn.functional.mse_loss(predictions, targets) * self.alpha
        kld_loss = (
            torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
            * self.beta
        )
        return mse_loss + kld_loss


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = Normal(torch.zeros_like(std), torch.ones_like(std)).sample()
    return mu + eps * std
