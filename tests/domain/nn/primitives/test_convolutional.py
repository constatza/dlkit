import pytest
import torch
from torch import nn

from dlkit.domain.nn.primitives.convolutional import DeconvolutionBlock1d


def test_deconv_block_output_shape(basic_input):
    m = DeconvolutionBlock1d(in_channels=4, out_channels=8, in_timesteps=16, kernel_size=3)
    assert m(basic_input).shape == (2, 8, 16)


def test_deconv_block_accepts_layer_norm(basic_input):
    m = DeconvolutionBlock1d(in_channels=4, out_channels=8, in_timesteps=16, normalize="layer")
    assert m(basic_input).shape == (2, 8, 16)


def test_deconv_block_accepts_batch_norm(basic_input):
    m = DeconvolutionBlock1d(in_channels=4, out_channels=8, in_timesteps=16, normalize="batch")
    m.eval()
    assert m(basic_input).shape == (2, 8, 16)


def test_deconv_block_activation_after_conv():
    """Activation must fire after conv, not before."""
    call_order = []
    m = DeconvolutionBlock1d(in_channels=4, out_channels=4, in_timesteps=16)
    m.conv1.register_forward_hook(lambda mod, inp, out: call_order.append("conv"))
    activation = m.activation
    assert isinstance(activation, nn.Module)
    activation.register_forward_hook(lambda mod, inp, out: call_order.append("act"))
    x = torch.randn(2, 4, 16)
    m(x)
    assert call_order == ["conv", "act"], f"Expected conv before act, got {call_order}"


def test_deconv_block_same_padding_stride_gt1_raises():
    """'same' padding with stride > 1 must raise ValueError."""
    with pytest.raises(ValueError, match='"same" padding'):
        DeconvolutionBlock1d(in_channels=4, out_channels=4, in_timesteps=16, stride=2)
