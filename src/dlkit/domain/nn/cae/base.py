import abc
from typing import Any

from torch import nn


class CAE(nn.Module):
    """Convolutional autoencoder base class.

    Provides abstract methods for encode and decode operations, and a standard
    forward pass that chains them together.
    """

    @abc.abstractmethod
    def encode(self, *args: Any, **kwargs: Any) -> Any:
        """Encode input to latent space.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Encoded latent representation.
        """
        ...

    @abc.abstractmethod
    def decode(self, *args: Any, **kwargs: Any) -> Any:
        """Decode latent representation back to original space.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Decoded output.
        """
        ...

    def forward(self, x: Any) -> Any:
        """Forward pass through autoencoder (encode -> decode).

        Args:
            x: Input data.

        Returns:
            Decoded output.
        """
        return self.decode(self.encode(x))
