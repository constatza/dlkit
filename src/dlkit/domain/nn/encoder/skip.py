from collections.abc import Callable, Sequence

from torch import Tensor, nn
from torch.nn import ModuleList

from dlkit.domain.nn.primitives.convolutional import ConvolutionBlock1d
from dlkit.domain.nn.primitives.skip import SkipConnection
from dlkit.domain.nn.types import NormalizerName


def _build_conv_stack(
    channels: Sequence[int],
    timesteps: Sequence[int],
    *,
    kernel_size: int,
    normalize: NormalizerName | None,
    activation: Callable[..., Tensor],
    dropout: float,
    dilation: int,
) -> ModuleList:
    """Build a stack of SkipConnection(ConvolutionBlock1d) layers.

    Args:
        channels: Channel counts per level (length == num_layers + 1).
        timesteps: Timestep counts per level (length == num_layers + 1).
        kernel_size: Convolution kernel size.
        normalize: Normalizer identifier forwarded to ConvolutionBlock1d.
        activation: Activation function.
        dropout: Dropout probability.
        dilation: Dilation base; incremented per layer inside the stack.

    Returns:
        ModuleList of SkipConnection modules, one per channel transition.
    """
    layers: ModuleList = ModuleList()
    for i in range(len(channels) - 1):
        layers.append(
            SkipConnection(
                ConvolutionBlock1d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    in_timesteps=timesteps[i],
                    kernel_size=kernel_size,
                    padding="same",
                    normalize=normalize,
                    dilation=i + 1,
                    activation=activation,
                    dropout=dropout,
                ),
                in_channels=channels[i],
                out_channels=channels[i + 1],
            )
        )
    return layers


class SkipEncoder1d(nn.Module):
    """Encoder that progressively compresses spatial and channel dimensions.

    Applies a stack of SkipConnection(ConvolutionBlock1d) layers, reducing
    the timestep dimension via interpolation after each layer.

    Args:
        channels: Channel counts at each level (length == num_layers + 1).
        timesteps: Timestep counts at each level (length == num_layers + 1).
        kernel_size: Convolution kernel size. Defaults to 3.
        activation: Activation function. Defaults to gelu.
        normalize: Normalizer identifier. Defaults to None.
        reduce: Spatial-reduction callable (e.g. ``nn.functional.interpolate``).
        dilation: Dilation base forwarded to conv stack. Defaults to 1.
        dropout: Dropout probability. Defaults to 0.0.
    """

    def __init__(
        self,
        channels: Sequence[int],
        timesteps: Sequence[int],
        kernel_size: int = 3,
        activation: Callable[..., Tensor] = nn.functional.gelu,
        normalize: NormalizerName | None = None,
        reduce: Callable[..., Tensor] = nn.functional.interpolate,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.timesteps = list(timesteps)
        self.channels = list(channels)
        self.reduce = reduce
        self.layers = _build_conv_stack(
            channels,
            timesteps,
            kernel_size=kernel_size,
            normalize=normalize,
            activation=activation,
            dropout=dropout,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode by applying skip-conv layers and downsampling between each.

        Args:
            x: Input tensor of shape ``(batch, channels[0], timesteps[0])``.

        Returns:
            Encoded tensor of shape ``(batch, channels[-1], timesteps[-1])``.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.reduce(x, size=self.timesteps[i + 1])
        return x


class SkipDecoder1d(nn.Module):
    """Decoder that progressively expands spatial and channel dimensions.

    Mirrors :class:`SkipEncoder1d` in structure but is an independent class
    (no inheritance relationship). Adds a final convolutional regression head.

    Args:
        channels: Channel counts at each level (length == num_layers + 1).
                  Typically the encoder channel schedule in reverse.
        timesteps: Timestep counts at each level (length == num_layers + 1).
                   Typically the encoder timestep schedule in reverse.
        kernel_size: Convolution kernel size. Defaults to 3.
        activation: Activation function. Defaults to gelu.
        normalize: Normalizer identifier. Defaults to None.
        reduce: Spatial-expansion callable (e.g. ``nn.functional.interpolate``).
        dilation: Dilation base forwarded to conv stack. Defaults to 1.
        dropout: Dropout probability. Defaults to 0.0.
    """

    def __init__(
        self,
        channels: Sequence[int],
        timesteps: Sequence[int],
        kernel_size: int = 3,
        activation: Callable[..., Tensor] = nn.functional.gelu,
        normalize: NormalizerName | None = None,
        reduce: Callable[..., Tensor] = nn.functional.interpolate,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.timesteps = list(timesteps)
        self.channels = list(channels)
        self.reduce = reduce
        self.layers = _build_conv_stack(
            channels,
            timesteps,
            kernel_size=kernel_size,
            normalize=normalize,
            activation=activation,
            dropout=dropout,
            dilation=dilation,
        )
        self.regression_layer = nn.Conv1d(
            in_channels=channels[-1],
            out_channels=channels[-1],
            kernel_size=kernel_size,
            padding="same",
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Decode by applying skip-conv layers, upsampling between each, then regression.

        Args:
            x: Input tensor of shape ``(batch, channels[0], timesteps[0])``.

        Returns:
            Decoded tensor of shape ``(batch, channels[-1], timesteps[-1])``.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.reduce(x, size=self.timesteps[i + 1])
        return self.regression_layer(x)
