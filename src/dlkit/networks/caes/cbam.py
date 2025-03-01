import torch
import torch.nn as nn


class ChannelAttention1D(nn.Module):
    """Channel Attention Module for 1D data."""

    def __init__(self, in_channels: int, out_channels: int):
        super(ChannelAttention1D, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.reduce_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.expand_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.global_avg_pool(x)  # Squeeze via avg pooling
        max_out = self.global_max_pool(x)  # Squeeze via max pooling
        # Shared reduction and expansion paths
        y = self.reduce_conv(avg_out) + self.reduce_conv(max_out)
        y = self.relu(y)
        y = self.sigmoid(self.expand_conv(y))
        return x * y  # Scale input by channel attention


class SpatialAttention1D(nn.Module):
    """Spatial Attention Module for 1D data."""

    def __init__(self, kernel_size: int):
        """
        Args:
            kernel_size (int): Kernel size for the spatial attention module.
        """
        super(SpatialAttention1D, self).__init__()
        assert (
            kernel_size % 2 == 1
        ), "Kernel size must be odd to maintain temporal dimensions."
        self.conv = nn.Conv1d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Mean across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max across channels
        y = torch.cat(
            [avg_out, max_out], dim=1
        )  # Concatenate along the channel dimension
        y = self.sigmoid(self.conv(y))  # Temporal attention
        return x * y  # Scale input by spatial attention


class CBAM1D(nn.Module):
    """CBAM Module for 1D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: int,
        spatial_attention_kernel_size: int,
        kernel_size: int,
        conv_padding: int,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels after convolution.
            reduction (int): Reduction ratio for the channel attention module.
            spatial_attention_kernel_size (int): Kernel size for the spatial attention module.
            kernel_size (int): Kernel size for the main convolution layer.
            conv_padding (int): Padding for the main convolution layer.
        """
        super(CBAM1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=conv_padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.channel_attention = ChannelAttention1D(out_channels, reduction)
        self.spatial_attention = SpatialAttention1D(spatial_attention_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # Convolution layer
        x = self.bn(x)  # Batch normalization
        x = self.relu(x)  # Activation
        x = self.channel_attention(x)  # Apply channel attention
        x = self.spatial_attention(x)  # Apply spatial attention
        return x
