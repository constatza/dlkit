from __future__ import annotations

from collections.abc import Callable

import torch

from dlkit.common.sources import InputShapes, OutputShapes
from dlkit.domain.nn.cae.base import CAE
from dlkit.domain.nn.contracts import (
    InputSpec as _InputSpec,
)
from dlkit.domain.nn.contracts import StandardEntryConsumer
from dlkit.domain.nn.encoder.latent import (
    TensorToVectorBlock,
    VectorToTensorBlock,
)
from dlkit.domain.nn.encoder.skip import SkipDecoder1d, SkipEncoder1d
from dlkit.domain.nn.types import ActivationName, NormalizerName
from dlkit.domain.nn.utils import build_channel_schedule, resolve_activation


class SkipCAE1d(StandardEntryConsumer, CAE):
    """1D Skip Connection Convolutional Autoencoder.

    Args:
        in_channels: Number of input channels.
        in_length: Length of input sequence.
        latent_channels: Number of latent channels.
        latent_size: Size of latent vector.
        latent_width: Width of latent feature (default: 1).
        num_layers: Number of encoder/decoder layers (default: 3).
        kernel_size: Convolution kernel size (default: 3).
        activation: Activation function (default: relu).
        normalize: Normalization type (default: None).
        dropout: Dropout probability (default: 0.0).
        transpose: Whether to transpose dimensions in latent encoding (default: False).
        dilation: Dilation for convolutions (default: 1).
    """

    class InputSpec(_InputSpec):
        pass

    _SHAPE_KWARG_NAMES: frozenset[str] = frozenset({"in_channels", "in_length"})

    @classmethod
    def _constructor_dims(
        cls,
        input_shapes: InputShapes,
        output_shapes: OutputShapes,
    ) -> dict[str, int]:
        """Derive channel and length dimensions from the first entry shape.

        Args:
            input_shapes: Mapping from feature entry name to its shape.
            output_shapes: Mapping from target entry name to its shape.

        Returns:
            Dict with ``in_channels`` and ``in_length``.

        Raises:
            ValueError: If input shape has fewer than 2 dimensions.
        """
        first_shape = next(iter(input_shapes.values()))
        if len(first_shape) < 2:
            raise ValueError(
                f"{cls.__name__} requires at least 2-D input (in_channels, in_length) "
                f"but entry has shape {first_shape}. "
                "Check that your feature entry produces a multi-dimensional tensor."
            )
        return {"in_channels": first_shape[0], "in_length": first_shape[1]}

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
        activation: ActivationName | Callable[[torch.Tensor], torch.Tensor] | None = None,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
        transpose: bool = False,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        resolved = resolve_activation(activation)
        channels = build_channel_schedule(in_channels, latent_channels, num_layers + 1)
        timesteps = build_channel_schedule(in_length, latent_width, num_layers + 1)

        self.encoder = SkipEncoder1d(
            channels,
            timesteps,
            kernel_size=kernel_size,
            normalize=normalize,
            activation=resolved,
            dropout=dropout,
            dilation=dilation,
        )

        reduce_dim = timesteps[-1] if transpose else channels[-1]
        self.feature_to_latent = TensorToVectorBlock(reduce_dim, latent_size, transpose=transpose)
        self.latent_to_feature = VectorToTensorBlock(latent_size, (channels[-1], timesteps[-1]))
        self.decoder = SkipDecoder1d(
            channels[::-1],
            timesteps[::-1],
            kernel_size=kernel_size,
            normalize=normalize,
            activation=resolved,
            dropout=dropout,
            dilation=dilation,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space.

        Args:
            x: Input tensor of shape (batch, in_channels, in_length).

        Returns:
            Latent representation tensor.
        """
        x = self.encoder(x)
        return self.feature_to_latent(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output space.

        Args:
            x: Latent tensor.

        Returns:
            Decoded output tensor.
        """
        x = self.latent_to_feature(x)
        return self.decoder(x)
