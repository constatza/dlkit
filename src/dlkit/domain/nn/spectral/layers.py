"""Reusable spectral / frequency-domain layers.

Both the Fourier-enhanced FFNN family and the Fourier Neural Operator (FNO)
build on these primitives.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SpectralConv1d(nn.Module):
    """Learnable spectral convolution for 1-D signals.

    Applies a learned complex weight matrix in the Fourier domain, keeping
    only the lowest ``n_modes`` frequency components.  All other modes are
    zeroed before the inverse FFT, which acts as a low-pass spectral filter
    whose cutoff is jointly optimised with the weights.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        n_modes: Number of Fourier modes to retain (per side of the
            real FFT spectrum, i.e. DC + ``n_modes - 1`` positive
            frequencies).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        n_modes: int,
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._n_modes = n_modes

        # Complex weights stored as two real tensors (real, imag).
        # Shape: (in_channels, out_channels, n_modes)
        scale = 1.0 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, n_modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, n_modes))

    @property
    def n_modes(self) -> int:
        """Number of retained Fourier modes.

        Returns:
            Mode count passed at construction.
        """
        return self._n_modes

    def _complex_multiply(self, x_ft: Tensor, n: int) -> Tensor:
        """Multiply truncated spectrum by the complex weight matrix.

        Args:
            x_ft: Complex tensor of shape
                ``(batch, in_channels, n)``.
            n: Number of active modes (may be less than ``self._n_modes``
               when the input is shorter than the weight matrix).

        Returns:
            Complex tensor of shape
                ``(batch, out_channels, n)``.
        """
        # Slice weights to match the actual number of active modes.
        w = torch.complex(self.weight_real[..., :n], self.weight_imag[..., :n])
        # einsum: (b, i, m), (i, o, m) -> (b, o, m)
        return torch.einsum("bim,iom->bom", x_ft, w)

    def forward(self, x: Tensor) -> Tensor:
        """Apply spectral convolution.

        Args:
            x: Real input of shape ``(batch, in_channels, length)``.

        Returns:
            Real output of shape ``(batch, out_channels, length)``.
        """
        length = x.shape[-1]
        x_ft = torch.fft.rfft(x, norm="ortho")  # (B, C, L//2+1)
        out_ft = torch.zeros(
            x.shape[0],
            self._out_channels,
            x_ft.shape[-1],
            dtype=x_ft.dtype,
            device=x.device,
        )
        n = min(self._n_modes, x_ft.shape[-1])
        out_ft[..., :n] = self._complex_multiply(x_ft[..., :n], n)
        return torch.fft.irfft(out_ft, n=length, norm="ortho")  # (B, C_out, L)


class FourierLayer(nn.Module):
    """Single FNO-style building block.

    Combines a spectral path (``SpectralConv1d``) with a local linear skip
    (``Conv1d`` with kernel size 1), sums the two paths and applies an
    activation.  Input and output channel counts are equal, so these blocks
    can be stacked freely.

    Reference: Li et al., "Fourier Neural Operator for Parametric Partial
    Differential Equations", ICLR 2021.

    Args:
        channels: Number of feature channels (in == out).
        n_modes: Number of Fourier modes kept in the spectral path.
        activation: Pointwise activation applied after the residual sum.
            Defaults to ``F.gelu``.
    """

    def __init__(
        self,
        *,
        channels: int,
        n_modes: int,
        activation: Callable[[Tensor], Tensor] = F.gelu,
    ) -> None:
        super().__init__()
        self._channels = channels
        self._n_modes = n_modes
        self.activation = activation

        self.spectral_conv = SpectralConv1d(
            in_channels=channels, out_channels=channels, n_modes=n_modes
        )
        self.local_conv = nn.Conv1d(channels, channels, kernel_size=1)

    @property
    def n_modes(self) -> int:
        """Number of Fourier modes in the spectral path.

        Returns:
            Value forwarded from the inner ``SpectralConv1d``.
        """
        return self._n_modes

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass combining spectral and local paths.

        Args:
            x: Input of shape ``(batch, channels, length)``.

        Returns:
            Output of shape ``(batch, channels, length)``.
        """
        return self.activation(self.spectral_conv(x) + self.local_conv(x))
