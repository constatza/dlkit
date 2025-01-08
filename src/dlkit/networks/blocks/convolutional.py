# Description: Convolutional blocks for use in neural networks.
import torch.nn as nn

class ConvSameTimesteps(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            batch_norm: bool = True,
    ):
        """
        A residual transposed convolutional block with upsampling.

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
            padding="same",
        )
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

class UpsampleTimesteps(nn.Module):
    def __init__(self, out_timesteps: int):
        """
        A convolutional block that changes the timesteps of the input.
        Args:
            out_timesteps:
        """
        super().__init__()
        self.out_timesteps = out_timesteps
        self.pooling = nn.Upsample(out_timesteps, mode="linear")

    def forward(self, x):
        x = self.pooling(x)
        return x

class DownsampleTimesteps(nn.Module):
    def __init__(self, out_timesteps: int):
        """
        A convolutional block that changes the timesteps of the input.
        Args:
            out_timesteps:
        """
        super().__init__()
        self.out_timesteps = out_timesteps
        self.pooling = nn.AdaptiveMaxPool1d(out_timesteps)

    def forward(self, x):
        x = self.pooling(x)
        return x