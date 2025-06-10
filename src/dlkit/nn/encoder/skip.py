from collections.abc import Callable
from torch import nn
from torch.nn import ModuleList
from collections.abc import Sequence

from dlkit.nn.primitives.convolutional import ConvolutionBlock1d
from dlkit.nn.primitives.skip import SkipConnection


class SkipEncoder1d(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        timesteps: Sequence[int],
        kernel_size: int = 3,
        activation: nn.functional = nn.functional.gelu,
        normalize: str | None = None,
        reduce: Callable = nn.functional.interpolate,
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
                        dilation=dilation,
                        activation=activation,
                        dropout=dropout,
                    ),
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                )
            )

        self.layers = layers

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.reduce(x, size=self.timesteps[i + 1])
        return x
