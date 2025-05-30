# Description: Convolutional blocks for use in neural networks.
import math

import torch.nn as nn


class ConvolutionBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_timesteps: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        padding: str | int = "same",
        groups: int = 1,
    ):
        """A residual transposed convolutional block with upsampling.

        Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - kernel_size (int): Kernel size for transposed convolutions.
        - batch_norm (bool): Whether to use batch normalization.
        """
        super().__init__()
        self.activation = nn.GELU()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
            groups=groups,
        )

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = output_size(in_timesteps, kernel_size, stride, padding)

    def forward(self, x):
        x = self.activation(x)
        x = self.conv1(x)
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
    ):
        """A residual transposed convolutional block with upsampling.

        Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - kernel_size (int): Kernel size for transposed convolutions.
        - batch_norm (bool): Whether to use batch normalization.
        """
        super().__init__()
        self.activation = nn.GELU()
        self.conv1 = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
            groups=groups,
            output_padding=output_padding,
        )

        self.layer_norm = nn.LayerNorm([in_channels, in_timesteps])
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.in_timesteps = in_timesteps

    def forward(self, x):
        # x = torch.nn.functional.interpolate(
        #     x,
        #     size=self.in_timesteps,
        #     mode="linear",
        # )
        # x = self.layer_norm(x)
        x = self.activation(x)
        x = self.conv1(x)
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
        padding (int): The amount of zero-padding applied.
        dilation (int, optional): The dilation rate. Defaults to 1.

    Returns:
        int: The computed output size.
    """
    if padding == "same":
        return input_size
    raw = math.floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    return int(raw)
