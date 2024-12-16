# tests/test_networks/test_caes/test_fixtures.py
import pytest
import torch
from dlkit.networks.caes.cbam import (
    ChannelAttention1D,
    SpatialAttention1D,
    CBAM1D,
)


@pytest.fixture(params=[(16, 4), (1, 1)])
def channel_attention_1d_data(request):
    """
    Fixture for generating data for ChannelAttention1D module.

    Parameters:
    - request: pytest request object to parametrize the fixture.

    Returns:
    - Tuple containing the input tensor and parameters.
    """
    in_channels, reduction = request.param
    x = torch.randn(8, in_channels, 100)
    return x, in_channels, reduction


@pytest.fixture(params=[7, 1])
def spatial_attention_1d_data(request):
    """
    Fixture for generating data for SpatialAttention1D module.

    Parameters:
    - request: pytest request object to parametrize the fixture.

    Returns:
    - Tuple containing the input tensor and kernel size.
    """
    kernel_size = request.param
    x = torch.randn(8, 16, 100)
    return x, kernel_size


@pytest.fixture(params=[(10, 16, 4, 7, 3, 1), (1, 1, 1, 1, 1, 0)])
def cbam_1d_data(request):
    """
    Fixture for generating data for CBAM1D module.

    Parameters:
    - request: pytest request object to parametrize the fixture.

    Returns:
    - Tuple containing the input tensor and parameters.
    """
    (
        in_channels,
        out_channels,
        reduction,
        kernel_size,
        conv_kernel_size,
        conv_padding,
    ) = request.param
    x = torch.randn(8, in_channels, 100)
    return (
        x,
        in_channels,
        out_channels,
        reduction,
        kernel_size,
        conv_kernel_size,
        conv_padding,
    )  # tests/test_networks/test_caes/test_cbam_block.py


def test_channel_attention_1d_forward(channel_attention_1d_data):
    """
    Test the forward pass of ChannelAttention1D module.

    Parameters:
    - channel_attention_1d_data: Fixture providing the input tensor and parameters.
    """
    x, in_channels, reduction = channel_attention_1d_data
    ca = ChannelAttention1D(in_channels, reduction)
    output = ca(x)
    assert output.shape == x.shape


def test_spatial_attention_1d_forward(spatial_attention_1d_data):
    """
    Test the forward pass of SpatialAttention1D module.

    Parameters:
    - spatial_attention_1d_data: Fixture providing the input tensor and kernel size.
    """
    x, kernel_size = spatial_attention_1d_data
    sa = SpatialAttention1D(kernel_size)
    output = sa(x)
    assert output.shape == x.shape


def test_cbam_1d_forward(cbam_1d_data):
    """
    Test the forward pass of CBAM1D module.

    Parameters:
    - cbam_1d_data: Fixture providing the input tensor and parameters.
    """
    (
        x,
        in_channels,
        out_channels,
        reduction,
        kernel_size,
        conv_kernel_size,
        conv_padding,
    ) = cbam_1d_data
    cbam = CBAM1D(
        in_channels,
        out_channels,
        reduction,
        kernel_size,
        conv_kernel_size,
        conv_padding,
    )
    output = cbam(x)
    assert output.shape == (8, out_channels, 100)
