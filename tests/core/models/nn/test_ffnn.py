"""Tests for feed-forward neural networks.

Tests FeedForwardNN, ConstantWidthFFNN, and LinearNetwork for shape
transformations, inheritance, and normalization/regularization.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dlkit.domain.nn.ffnn.linear import LinearNetwork
from dlkit.domain.nn.ffnn.simple import ConstantWidthFFNN, FeedForwardNN


@pytest.fixture
def ffnn() -> FeedForwardNN:
    """Basic FFNN with 2 hidden layers."""
    return FeedForwardNN(in_features=2, out_features=2, layers=[4, 4])


@pytest.fixture
def constant_ffnn() -> ConstantWidthFFNN:
    """FFNN with constant width hidden layers."""
    return ConstantWidthFFNN(in_features=2, out_features=2, hidden_size=4, num_layers=3)


@pytest.fixture
def linear_net() -> LinearNetwork:
    """Simple linear network."""
    return LinearNetwork(in_features=2, out_features=2)


class TestFeedForwardNN:
    """Tests for FeedForwardNN."""

    def test_output_shape(self, ffnn: FeedForwardNN, dense_input: torch.Tensor) -> None:
        """Output should have shape (batch, out_features)."""
        assert ffnn(dense_input).shape == (dense_input.shape[0], 2)

    def test_is_dlkit_model(self, ffnn: FeedForwardNN) -> None:
        """FeedForwardNN should be instance of nn.Module."""
        assert isinstance(ffnn, nn.Module)

    def test_is_nn_module(self, ffnn: FeedForwardNN) -> None:
        """FeedForwardNN should be instance of nn.Module."""
        assert isinstance(ffnn, nn.Module)

    def test_single_hidden_layer(self, dense_input: torch.Tensor) -> None:
        """FFNN with single hidden layer should work."""
        m = FeedForwardNN(in_features=2, out_features=2, layers=[4])
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_many_hidden_layers(self, dense_input: torch.Tensor) -> None:
        """FFNN with many layers should work."""
        m = FeedForwardNN(in_features=2, out_features=2, layers=[4, 8, 4, 2])
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_with_batch_norm(self, dense_input: torch.Tensor) -> None:
        """FFNN with batch normalization should work."""
        m = FeedForwardNN(in_features=2, out_features=2, layers=[4, 4], normalize="batch")
        m.eval()
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_with_layer_norm(self, dense_input: torch.Tensor) -> None:
        """FFNN with layer normalization should work."""
        m = FeedForwardNN(in_features=2, out_features=2, layers=[4, 4], normalize="layer")
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_with_dropout(self, dense_input: torch.Tensor) -> None:
        """FFNN with dropout should work."""
        m = FeedForwardNN(in_features=2, out_features=2, layers=[4, 4], dropout=0.2)
        m.eval()
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_has_parameters(self, ffnn: FeedForwardNN) -> None:
        """FFNN should have trainable parameters."""
        assert len(list(ffnn.parameters())) > 0

    def test_has_embedding_layer(self, ffnn: FeedForwardNN) -> None:
        """FFNN should have an embedding_layer."""
        assert hasattr(ffnn, "embedding_layer")
        assert isinstance(ffnn.embedding_layer, nn.Linear)
        assert ffnn.embedding_layer.in_features == 2
        assert ffnn.embedding_layer.out_features == 4

    def test_has_regression_layer(self, ffnn: FeedForwardNN) -> None:
        """FFNN should have a regression_layer."""
        assert hasattr(ffnn, "regression_layer")
        assert isinstance(ffnn.regression_layer, nn.Linear)
        assert ffnn.regression_layer.in_features == 4
        assert ffnn.regression_layer.out_features == 2

    def test_has_hidden_layers(self, ffnn: FeedForwardNN) -> None:
        """FFNN should have layers ModuleList."""
        assert hasattr(ffnn, "layers")
        assert isinstance(ffnn.layers, nn.ModuleList)
        assert len(ffnn.layers) == 1  # [4 → 4]

    def test_num_layers_stored(self, ffnn: FeedForwardNN) -> None:
        """FFNN should store num_layers."""
        assert ffnn.num_layers == 2  # len([4, 4])

    def test_activation_stored(self, ffnn: FeedForwardNN) -> None:
        """FFNN should store activation function."""
        assert hasattr(ffnn, "activation")
        assert callable(ffnn.activation)

    def test_different_widths(self, dense_input: torch.Tensor) -> None:
        """FFNN should support varying layer widths."""
        m = FeedForwardNN(in_features=2, out_features=2, layers=[8, 4, 6])
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_gradient_flow(self, dense_input: torch.Tensor) -> None:
        """Gradients should flow through all layers."""
        ffnn = FeedForwardNN(in_features=2, out_features=2, layers=[4, 4])
        out = ffnn(dense_input)
        loss = out.sum()
        loss.backward()
        # All parameters should have gradients
        for param in ffnn.parameters():
            assert param.grad is not None


class TestConstantWidthFFNN:
    """Tests for ConstantWidthFFNN."""

    def test_output_shape(
        self, constant_ffnn: ConstantWidthFFNN, dense_input: torch.Tensor
    ) -> None:
        """Output should have shape (batch, out_features)."""
        assert constant_ffnn(dense_input).shape == (dense_input.shape[0], 2)

    def test_is_ffnn(self, constant_ffnn: ConstantWidthFFNN) -> None:
        """ConstantWidthFFNN should be instance of FeedForwardNN."""
        assert isinstance(constant_ffnn, FeedForwardNN)

    def test_is_dlkit_model(self, constant_ffnn: ConstantWidthFFNN) -> None:
        """ConstantWidthFFNN should be instance of nn.Module."""
        assert isinstance(constant_ffnn, nn.Module)

    def test_is_nn_module(self, constant_ffnn: ConstantWidthFFNN) -> None:
        """ConstantWidthFFNN should be instance of nn.Module."""
        assert isinstance(constant_ffnn, nn.Module)

    def test_all_hidden_layers_same_width(self) -> None:
        """All hidden layers should have width = hidden_size."""
        m = ConstantWidthFFNN(in_features=2, out_features=2, hidden_size=8, num_layers=4)
        # Check that layers were created with constant width
        # First layer: 2 → 8, then 8 → 8 × (num_layers-1)
        assert m.embedding_layer.out_features == 8
        assert m.regression_layer.in_features == 8

    def test_zero_layers_raises(self) -> None:
        """Zero hidden layers should raise ValueError."""
        with pytest.raises(ValueError):
            ConstantWidthFFNN(in_features=2, out_features=2, hidden_size=4, num_layers=0)

    def test_single_hidden_layer(self, dense_input: torch.Tensor) -> None:
        """ConstantWidthFFNN with single layer should work."""
        m = ConstantWidthFFNN(in_features=2, out_features=2, hidden_size=4, num_layers=1)
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_many_hidden_layers(self, dense_input: torch.Tensor) -> None:
        """ConstantWidthFFNN with many layers should work."""
        m = ConstantWidthFFNN(in_features=2, out_features=2, hidden_size=4, num_layers=5)
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_has_parameters(self, constant_ffnn: ConstantWidthFFNN) -> None:
        """ConstantWidthFFNN should have trainable parameters."""
        assert len(list(constant_ffnn.parameters())) > 0


class TestLinearNetwork:
    """Tests for LinearNetwork."""

    def test_output_shape(self, linear_net: LinearNetwork, dense_input: torch.Tensor) -> None:
        """Output should have shape (batch, out_features)."""
        assert linear_net(dense_input).shape == (dense_input.shape[0], 2)

    def test_is_dlkit_model(self, linear_net: LinearNetwork) -> None:
        """LinearNetwork should be instance of nn.Module."""
        assert isinstance(linear_net, nn.Module)

    def test_is_nn_module(self, linear_net: LinearNetwork) -> None:
        """LinearNetwork should be instance of nn.Module."""
        assert isinstance(linear_net, nn.Module)

    def test_with_batch_norm(self, dense_input: torch.Tensor) -> None:
        """LinearNetwork with batch norm should work."""
        m = LinearNetwork(in_features=2, out_features=2, normalize="batch")
        m.eval()
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_with_layer_norm(self, dense_input: torch.Tensor) -> None:
        """LinearNetwork with layer norm should work."""
        m = LinearNetwork(in_features=2, out_features=2, normalize="layer")
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_without_normalization(self, dense_input: torch.Tensor) -> None:
        """LinearNetwork without normalization should work."""
        m = LinearNetwork(in_features=2, out_features=2, normalize=None)
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_no_bias(self, dense_input: torch.Tensor) -> None:
        """LinearNetwork without bias should work."""
        m = LinearNetwork(in_features=2, out_features=2, bias=False)
        assert m(dense_input).shape == (dense_input.shape[0], 2)
        assert m.linear.bias is None

    def test_with_bias(self) -> None:
        """LinearNetwork with bias should have bias parameter."""
        m = LinearNetwork(in_features=2, out_features=2, bias=True)
        assert m.linear.bias is not None

    def test_has_linear_layer(self, linear_net: LinearNetwork) -> None:
        """LinearNetwork should have a linear layer."""
        assert hasattr(linear_net, "linear")
        assert isinstance(linear_net.linear, nn.Linear)
        assert linear_net.linear.in_features == 2
        assert linear_net.linear.out_features == 2

    def test_has_norm_layer(self, linear_net: LinearNetwork) -> None:
        """LinearNetwork should have a norm layer."""
        assert hasattr(linear_net, "norm")
        assert isinstance(linear_net.norm, nn.Module)

    def test_has_parameters(self, linear_net: LinearNetwork) -> None:
        """LinearNetwork should have trainable parameters."""
        assert len(list(linear_net.parameters())) > 0

    def test_different_in_out_features(self, batch_size: int) -> None:
        """LinearNetwork should support different input/output sizes."""
        m = LinearNetwork(in_features=8, out_features=4)
        x = torch.randn(batch_size, 8)
        assert m(x).shape == (batch_size, 4)

    def test_gradient_flow(self, dense_input: torch.Tensor) -> None:
        """Gradients should flow through network."""
        linear_net = LinearNetwork(in_features=2, out_features=2)
        out = linear_net(dense_input)
        loss = out.sum()
        loss.backward()
        for param in linear_net.parameters():
            assert param.grad is not None
