"""Fourier Neural Operator for 1-D spatial domains.

Reference: Li et al., "Fourier Neural Operator for Parametric Partial
Differential Equations", ICLR 2021. https://arxiv.org/abs/2010.08895
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn.functional as F
from torch import Tensor, nn

from dlkit.domain.nn.operators.base import GridOperatorBase
from dlkit.domain.nn.spectral.layers import FourierLayer


class FourierNeuralOperator1d(GridOperatorBase):
    """Fourier Neural Operator for 1-D spatial functions.

    Learns a mapping ``G : U → V`` between function spaces discretised on
    a regular 1-D grid.  The operator is discretisation-invariant: a model
    trained on a coarse grid can be evaluated at a finer resolution.

    Inherits the lifting → body → projection scaffold from
    ``GridOperatorBase``.  The body is an ``nn.Sequential`` of
    ``FourierLayer`` blocks.

    To build an operator with a custom body (e.g. wavelet layers), subclass
    ``GridOperatorBase`` directly and supply your own body::

        body = nn.Sequential(*[WaveletLayer(channels=64) for _ in range(4)])
        op = GridOperatorBase(body=body, in_channels=2, out_channels=1, width=64)

    Args:
        in_channels: Number of input function channels (including any
            appended spatial grid coordinates).
        out_channels: Number of output function channels.
        width: Latent channel width used throughout the FNO body.
        n_modes: Number of Fourier modes retained per layer.
        n_layers: Number of ``FourierLayer`` blocks in the body.
        activation: Pointwise activation used inside each Fourier layer.
            Defaults to ``F.gelu``.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        n_modes: int = 16,
        n_layers: int = 4,
        activation: Callable[[Tensor], Tensor] = F.gelu,
    ) -> None:
        body: nn.Module = nn.Sequential(
            *[
                FourierLayer(channels=width, n_modes=n_modes, activation=activation)
                for _ in range(n_layers)
            ]
        )
        super().__init__(
            body=body,
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
        )
