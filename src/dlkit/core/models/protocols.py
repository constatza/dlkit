"""Structural protocols for DLKit model capabilities.

This module provides Protocol classes that define capability interfaces for
model types. These are structural (not inheritance-based) — models just need
to implement the right methods to satisfy the protocol.

Use isinstance(model, IAutoencoder) at predict time to populate latents.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class IAutoencoder(Protocol):
    """Protocol for encoder-decoder models (CAE, VAE, etc.).

    Models satisfying this protocol expose encode() and decode() methods
    that are used by the wrapper to populate latents in predict_step.
    """

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent representation.

        Args:
            x: Input tensor.

        Returns:
            Latent representation tensor.
        """
        ...

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation back to input space.

        Args:
            z: Latent representation tensor.

        Returns:
            Reconstructed tensor.
        """
        ...


@runtime_checkable
class IVariationalAutoencoder(Protocol):
    """Protocol for variational autoencoder models (VAE).

    VAE models return (mu, log_var) from encode and support reparameterization.
    """

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input to (mu, log_var) distribution parameters.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (mu, log_var) tensors.
        """
        ...

    def decode(self, z: Tensor) -> Tensor:
        """Decode sampled latent to input space.

        Args:
            z: Sampled latent tensor.

        Returns:
            Reconstructed tensor.
        """
        ...

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample from latent distribution using reparameterization trick.

        Args:
            mu: Mean of the latent distribution.
            log_var: Log variance of the latent distribution.

        Returns:
            Sampled latent tensor.
        """
        ...
