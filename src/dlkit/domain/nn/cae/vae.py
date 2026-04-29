"""Variational Autoencoder for 1D convolutional data."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.distributions.normal import Normal

from dlkit.domain.nn.encoder.latent import TensorToVectorBlock, VectorToTensorBlock
from dlkit.domain.nn.encoder.skip import SkipDecoder1d, SkipEncoder1d
from dlkit.domain.nn.types import NormalizerName
from dlkit.domain.nn.utils import build_channel_schedule

if TYPE_CHECKING:
    from dlkit.common.shapes import ShapeSummary


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample from the latent distribution using the reparameterization trick.

    Args:
        mu: Mean of the latent distribution.
        logvar: Log-variance of the latent distribution.

    Returns:
        Sampled latent vector ``z = mu + eps * std``.
    """
    std = torch.exp(0.5 * logvar)
    eps = Normal(torch.zeros_like(std), torch.ones_like(std)).sample()
    return mu + eps * std


def vae_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute the VAE loss: reconstruction loss + KL divergence.

    Args:
        predictions: Reconstructed output tensor.
        targets: Ground-truth input tensor.
        mu: Latent mean from the encoder.
        logvar: Latent log-variance from the encoder.
        alpha: Weight for the reconstruction (MSE) term. Defaults to 1.0.
        beta: Weight for the KL divergence term. Defaults to 0.1.

    Returns:
        Scalar loss tensor.
    """
    mse = nn.functional.mse_loss(predictions, targets) * alpha
    kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0) * beta
    return mse + kld


class VAE1d(nn.Module):
    """Variational Autoencoder for 1D convolutional data.

    Encodes input to a (mu, logvar) latent distribution, samples via
    reparameterization, and decodes back to the original shape.

    Use :func:`vae_loss` to compute the combined reconstruction + KL loss.

    Args:
        in_channels: Number of input channels.
        in_length: Length of the input sequence.
        latent_channels: Number of channels in the bottleneck feature map.
        latent_size: Dimension of the latent vector (mu / logvar).
        latent_width: Spatial width of the bottleneck feature map. Defaults to 1.
        num_layers: Number of encoder/decoder layers. Defaults to 3.
        kernel_size: Convolution kernel size. Defaults to 3.
        activation: Activation function. Defaults to gelu.
        normalize: Normalizer identifier. Defaults to None.
        dropout: Dropout probability. Defaults to 0.0.
        scale_of_latent: Multiplier for the intermediate projection size. Defaults to 4.
        alpha: Reconstruction loss weight (stored for convenience). Defaults to 1.0.
        beta: KL divergence loss weight (stored for convenience). Defaults to 0.1.
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
        scale_of_latent: int = 4,
        alpha: float = 1.0,
        beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.in_channels = in_channels
        self.in_length = in_length
        self.latent_size = latent_size

        channels = build_channel_schedule(in_channels, latent_channels, num_layers + 1)
        timesteps = build_channel_schedule(in_length, latent_width, num_layers + 1)

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
        self.mu_layer = nn.Linear(scale_of_latent * latent_size, latent_size)
        self.logvar_layer = nn.Linear(scale_of_latent * latent_size, latent_size)

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> VAE1d:
        """Build the VAE from a 1-D convolutional shape summary."""
        return cls(
            in_channels=shape.in_channels,
            in_length=shape.in_length,
            **kwargs,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor of shape ``(batch, in_channels, in_length)``.

        Returns:
            Tuple ``(mu, logvar)`` each of shape ``(batch, latent_size)``.
        """
        h = self.vectorize(self.encoder(x))
        return self.mu_layer(h), self.logvar_layer(h)

    def decode(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample from latent and decode to output space.

        Args:
            mu: Latent mean of shape ``(batch, latent_size)``.
            logvar: Latent log-variance of shape ``(batch, latent_size)``.

        Returns:
            Tuple ``(reconstruction, mu, logvar)``.
        """
        z = reparameterize(mu, logvar)
        y = self.decoder(self.tensorize(z))
        return y, mu, logvar

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward pass: encode → reparameterize → decode.

        Args:
            x: Input tensor of shape ``(batch, in_channels, in_length)``.

        Returns:
            Tuple ``(reconstruction, mu, logvar)``.
        """
        mu, logvar = self.encode(x)
        return self.decode(mu, logvar)
