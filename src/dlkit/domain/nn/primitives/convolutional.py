# Description: Convolutional blocks for use in neural networks.
import math
from collections.abc import Callable

import torch
from torch import nn

from dlkit.domain.nn.types import ActivationName, NormalizerName
from dlkit.domain.nn.utils import make_norm_layer, resolve_activation


class ConvolutionBlock1d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        in_timesteps: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str | int = "same",
        activation: ActivationName | Callable[[torch.Tensor], torch.Tensor] | None = None,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
        dilation: int = 1,
        groups: int = 1,
    ):
        """A 1-D causal convolutional block applying Norm → Conv → Act → Dropout.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            in_timesteps (int): Number of input timesteps (used for LayerNorm).
            kernel_size (int, optional): Convolution kernel size. Defaults to 3.
            stride (int, optional): Convolution stride. Defaults to 1.
            padding (str | int, optional): Padding mode or amount. Defaults to "same".
            activation (ActivationName | Callable[[torch.Tensor], torch.Tensor] | None, optional): Activation function or name. Defaults to relu.
            normalize (NormalizerName | None, optional): Normalization type
                (``"layer"`` or ``"batch"``). Defaults to None.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            dilation (int, optional): Dilation rate. Defaults to 1.
            groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
        """
        super().__init__()
        self.activation = resolve_activation(activation)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
            groups=groups,
        )
        self.dropout = nn.Dropout1d(dropout) if dropout > 0.0 else nn.Identity()
        self.norm = make_norm_layer(normalize, in_channels, in_timesteps)

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.in_timesteps = in_timesteps

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class DeconvolutionBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_timesteps: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        padding: str | int = "same",
        output_padding: int = 0,
        groups: int = 1,
        activation: ActivationName | Callable[[torch.Tensor], torch.Tensor] | None = None,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
    ):
        """A 1-D transposed convolutional block applying Norm → ConvTranspose → Act → Dropout.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            in_timesteps (int): Number of input timesteps (used for LayerNorm).
            kernel_size (int, optional): Kernel size for transposed convolutions. Defaults to 3.
            dilation (int, optional): Dilation rate. Defaults to 1.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (str | int, optional): Padding mode or amount. ``"same"`` maps to
                ``effective_padding = (kernel_size - 1) // 2 * dilation`` so that the output
                length equals the input length when ``stride=1``. Defaults to ``"same"``.
            output_padding (int, optional): Additional size added to one side of the output.
                Defaults to 0.
            groups (int, optional): Number of groups for grouped convolutions. Defaults to 1.
            activation (ActivationName | Callable[[torch.Tensor], torch.Tensor] | None, optional): Activation function or name. Defaults to relu.
            normalize (NormalizerName | None, optional): Normalization type
                (``"layer"`` or ``"batch"``). Applied before the convolution. Defaults to None.
            dropout (float, optional): Dropout probability applied after activation.
                Defaults to 0.0.
        """
        super().__init__()
        if padding == "same" and stride != 1:
            raise ValueError(
                '"same" padding for DeconvolutionBlock1d is only supported when stride=1.'
            )
        self.activation = resolve_activation(activation)
        effective_padding: int = (
            (kernel_size - 1) // 2 * dilation if padding == "same" else int(padding)
        )
        self.conv1 = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=effective_padding,
            stride=stride,
            groups=groups,
            output_padding=output_padding,
        )
        self.norm = make_norm_layer(normalize, in_channels, in_timesteps)
        self.dropout = nn.Dropout1d(dropout) if dropout > 0.0 else nn.Identity()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.in_timesteps = in_timesteps

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


def output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int | str,
    dilation: int = 1,
) -> int:
    """Compute the size of the output dimension of a convolution.

    Args:
        input_size (int): The size (height or width) of the input.
        kernel_size (int): The size of the convolution kernel.
        stride (int): The stride of the convolution.
        padding (int | str): The amount of zero-padding applied or padding mode.
        dilation (int, optional): The dilation rate. Defaults to 1.

    Returns:
        int: The computed output size.
    """
    match padding:
        case "same":
            return input_size
        case str():
            raise ValueError(f"Unsupported padding string {padding!r} for output size calculation")
        case _:
            raw = math.floor(
                (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
            )
            return int(raw)
