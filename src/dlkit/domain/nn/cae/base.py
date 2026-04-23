import abc

from torch import Tensor, nn


class CAE(nn.Module):
    """Convolutional autoencoder base class.

    Provides abstract methods for encode and decode operations, and a standard
    forward pass that chains them together.
    """

    @abc.abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent space.

        Args:
            x: Input tensor.

        Returns:
            Encoded latent representation.
        """
        ...

    @abc.abstractmethod
    def decode(self, x: Tensor) -> Tensor:
        """Decode latent representation back to original space.

        Args:
            x: Latent tensor.

        Returns:
            Decoded output.
        """
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through autoencoder (encode -> decode).

        Args:
            x: Input data.

        Returns:
            Decoded output.
        """
        return self.decode(self.encode(x))
