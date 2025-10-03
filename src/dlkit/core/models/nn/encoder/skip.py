from collections.abc import Callable, Sequence
from typing import Literal

from torch import Tensor, nn
from torch.nn import ModuleList

from dlkit.core.datatypes.networks import NormalizerName
from dlkit.core.models.nn.primitives.convolutional import ConvolutionBlock1d
from dlkit.core.models.nn.primitives.skip import SkipConnection


class SkipEncoder1d(nn.Module):
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
    ):
        """Complete encoder that compresses the input into a latent vector.

        Parameters:
        - latent_dim (int): Dimension of the latent vector.
        - channels (List[int]): List of channels for each layer.
        - kernel_size (int): Kernel size for convolutions.
        - timesteps (List[int]): List of timesteps for adaptive pooling at each layer.
        """
        super().__init__()

        self.timesteps = timesteps
        self.channels = channels
        self.reduce = reduce
        layers = ModuleList()

        if normalize in (None, "none"):
            norm_choice: Literal["layer", "batch", "instance"] | None = None
        else:
            norm_choice = normalize

        for i in range(len(channels) - 1):
            layers.append(
                SkipConnection(
                    ConvolutionBlock1d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        in_timesteps=timesteps[i],
                        kernel_size=kernel_size,
                        padding="same",
                        normalize=norm_choice,
                        dilation=i + 1,
                        activation=activation,
                        dropout=dropout,
                    ),
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                )
            )

        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.reduce(x, size=self.timesteps[i + 1])
        return x


class SkipDecoder1d(SkipEncoder1d):
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
    ):
        """Complete encoder that compresses the input into a latent vector.

        Parameters:
        - latent_dim (int): Dimension of the latent vector.
        - channels (List[int]): List of channels for each layer.
        - kernel_size (int): Kernel size for convolutions.
        - timesteps (List[int]): List of timesteps for adaptive pooling at each layer.
        """
        super().__init__(
            channels=channels,
            timesteps=timesteps,
            kernel_size=kernel_size,
            activation=activation,
            normalize=normalize,
            reduce=reduce,
            dilation=dilation,
            dropout=dropout,
        )
        self.regression_layer = nn.Conv1d(
            in_channels=channels[-1],
            out_channels=channels[-1],
            kernel_size=kernel_size,
            padding="same",
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = super().forward(x)
        x = self.regression_layer(x)
        return x
