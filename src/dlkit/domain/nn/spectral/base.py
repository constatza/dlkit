"""Protocols for spectral / frequency-domain layers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class ISpectralLayer(Protocol):
    """Layer that maps inputs through a learned spectral transform.

    Implementors operate in the frequency domain: they truncate the Fourier
    spectrum to ``n_modes`` coefficients, apply a learned weight, and return
    a spatial-domain output of the same length as the input.
    """

    @property
    def n_modes(self) -> int:
        """Number of frequency modes retained after truncation.

        Returns:
            Count of kept Fourier modes.
        """
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Apply the spectral transform.

        Args:
            x: Input tensor of shape ``(batch, channels, length)``.

        Returns:
            Output tensor of shape ``(batch, channels, length)``.
        """
        ...
