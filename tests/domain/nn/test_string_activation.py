"""Verify that model constructors accept activation as a string name."""

from typing import cast

import pytest
import torch

from dlkit.common.types import ActivationName
from dlkit.domain.nn.encoder.skip import SkipEncoder1d
from dlkit.domain.nn.ffnn.residual import FFNN
from dlkit.domain.nn.primitives.convolutional import ConvolutionBlock1d
from dlkit.domain.nn.primitives.dense import DenseBlock


@pytest.fixture
def batch() -> torch.Tensor:
    return torch.randn(3, 8)


@pytest.fixture
def conv_batch() -> torch.Tensor:
    return torch.randn(3, 4, 16)


@pytest.mark.parametrize("name", ["relu", "gelu", "silu", "tanh", "sigmoid", "leaky_relu", "none"])
def test_dense_block_accepts_string_activation(name: str, batch: torch.Tensor) -> None:
    block = DenseBlock(in_features=8, out_features=8, activation=cast(ActivationName, name))
    out = block(batch)
    assert out.shape == (3, 8)


@pytest.mark.parametrize("name", ["relu", "gelu", "silu"])
def test_ffnn_accepts_string_activation(name: str, batch: torch.Tensor) -> None:
    model = FFNN(in_features=8, out_features=4, num_layers=2, activation=cast(ActivationName, name))
    out = model(batch)
    assert out.shape == (3, 4)


@pytest.mark.parametrize("name", ["relu", "gelu"])
def test_convolution_block_accepts_string_activation(name: str, conv_batch: torch.Tensor) -> None:
    block = ConvolutionBlock1d(
        in_channels=4, out_channels=4, in_timesteps=16, activation=cast(ActivationName, name)
    )
    out = block(conv_batch)
    assert out.shape == (3, 4, 16)


@pytest.mark.parametrize("name", ["relu", "gelu"])
def test_skip_encoder_accepts_string_activation(name: str, conv_batch: torch.Tensor) -> None:
    enc = SkipEncoder1d(channels=[4, 8], timesteps=[16, 8], activation=cast(ActivationName, name))
    out = enc(conv_batch)
    assert out.shape[0] == 3
