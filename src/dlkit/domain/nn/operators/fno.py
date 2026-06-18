"""Fourier Neural Operator for 1-D spatial domains."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

from dlkit.common.shapes import InputShapes, OutputShapes
from dlkit.domain.nn.contracts import (
    InputSpec as _InputSpec,
)
from dlkit.domain.nn.contracts import StandardEntryConsumer
from dlkit.domain.nn.operators.base import GridOperatorBase
from dlkit.domain.nn.spectral.layers import FourierLayer
from dlkit.domain.nn.types import ActivationName
from dlkit.domain.nn.utils import resolve_activation


class FourierNeuralOperator1d(StandardEntryConsumer, GridOperatorBase):
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

    class InputSpec(_InputSpec):
        pass

    _SHAPE_KWARG_NAMES: frozenset[str] = frozenset({"in_channels", "out_channels"})

    @classmethod
    def _constructor_dims(
        cls,
        input_shapes: InputShapes,
        output_shapes: OutputShapes,
    ) -> dict[str, int]:
        """Derive channel dimensions from entry shapes.

        Expects 2-D input of shape ``(channels, length)``.

        Args:
            input_shapes: Mapping from feature entry name to its shape.
            output_shapes: Mapping from target entry name to its shape.

        Returns:
            Dict with ``in_channels`` and ``out_channels``.

        Raises:
            ValueError: If input shape has fewer than 2 dimensions.
        """
        in_shape = next(iter(input_shapes.values()))
        if len(in_shape) < 2:
            raise ValueError(
                f"{cls.__name__} requires a 2-D input shape (channels, length) "
                f"but got shape {in_shape}. "
                "Check that your feature entry produces a (C, L) tensor."
            )
        return {
            "in_channels": in_shape[0],
            "out_channels": next(iter(output_shapes.values()))[0],
        }

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
