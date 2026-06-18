"""Fourier Neural Operator for 1-D spatial domains."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self

from torch import Tensor, nn

from dlkit.common.sources import InputShapes, OutputShapes
from dlkit.domain.nn.operators.base import GridOperatorBase
from dlkit.domain.nn.spectral.layers import FourierLayer
from dlkit.domain.nn.types import ActivationName
from dlkit.domain.nn.utils import resolve_activation


class FourierNeuralOperator1d(GridOperatorBase):
    """Fourier Neural Operator for 1-D spatial functions.

    Input/output dimensions:
        input: ``(batch, in_channels, length)``
        output: ``(batch, out_channels, length)``

    Architecture dimensions:
        lifting: ``(batch, in_channels, length) -> (batch, width, length)``
        body: ``(batch, width, length) -> (batch, width, length)``
        projection: ``(batch, width, length) -> (batch, out_channels, length)``

    Constructor dimensions:
        ``in_channels``, ``out_channels``, ``width``, ``n_modes``, ``n_layers``
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        n_modes: int = 16,
        n_layers: int = 4,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        resolved = resolve_activation(activation)
        body: nn.Module = nn.Sequential(
            *[
                FourierLayer(channels=width, n_modes=n_modes, activation=resolved)
                for _ in range(n_layers)
            ]
        )
        super().__init__(
            body=body,
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
        )

    @classmethod
    def from_entries(
        cls, input_shapes: InputShapes, output_shapes: OutputShapes, **kwargs: Any
    ) -> Self:
        """Build the operator from dataset entry shapes.

        Args:
            input_shapes: Mapping from input name to its per-sample shape.
            output_shapes: Mapping from output name to its per-sample shape.
            **kwargs: Additional constructor arguments.

        Returns:
            Constructed operator instance.
        """
        in_channels = next(iter(input_shapes.values()))[0]
        out_channels = next(iter(output_shapes.values()))[0]
        return cls(in_channels=in_channels, out_channels=out_channels, **kwargs)
