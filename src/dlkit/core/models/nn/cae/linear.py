import torch
from dlkit.core.models.nn.cae.base import CAE
from dlkit.core.models.nn.cae.skip1d import SkipCAE1d


class LinearCAE1d(CAE):
    """1D Linear Convolutional Autoencoder.

    A simplified CAE that uses an identity activation (no non-linearity) for
    the SkipCAE1d backend.

    Args:
        in_channels: Number of input channels.
        in_length: Length of input sequence.
        latent_channels: Number of latent channels (default: 5).
        latent_width: Width of latent feature (default: 10).
        latent_size: Size of latent vector (default: 10).
        num_layers: Number of encoder/decoder layers (default: 3).
        kernel_size: Convolution kernel size (default: 3).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        in_length: int,
        latent_channels: int = 5,
        latent_width: int = 10,
        latent_size: int = 10,
        num_layers: int = 3,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self._impl = SkipCAE1d(
            in_channels=in_channels,
            in_length=in_length,
            latent_channels=latent_channels,
            latent_width=latent_width,
            latent_size=latent_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            activation=lambda x: x,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space.

        Args:
            x: Input tensor of shape (batch, in_channels, in_length).

        Returns:
            Latent representation tensor.
        """
        return self._impl.encode(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output space.

        Args:
            x: Latent tensor.

        Returns:
            Decoded output tensor.
        """
        return self._impl.decode(x)
