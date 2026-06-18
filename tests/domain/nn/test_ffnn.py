"""Tests for feed-forward neural networks.

Tests VarWidthFFNN, FFNN, and LinearNetwork for shape
transformations, inheritance, and normalization/regularization.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dlkit.domain.nn.ffnn.constrained import _resolve_hidden_size
from dlkit.domain.nn.ffnn.linear import (
    FactorizedLinearNetwork,
    LinearNetwork,
    SPDFactorizedLinearNetwork,
    SPDLinearNetwork,
    SymmetricFactorizedLinearNetwork,
    SymmetricLinearNetwork,
)
from dlkit.domain.nn.ffnn.residual import FFNN, VarWidthFFNN
from dlkit.domain.nn.primitives.parametrized_layers import (
    FactorizedLinear,
    SPDFactorizedLinear,
    SPDLinear,
    SymmetricFactorizedLinear,
    SymmetricLinear,
)


@pytest.fixture
def ffnn() -> VarWidthFFNN:
    """Basic FFNN with 2 hidden layers."""
    return VarWidthFFNN(in_features=2, out_features=2, layers=[4, 4])


@pytest.fixture
def constant_ffnn() -> FFNN:
    """FFNN with constant width hidden layers."""
    return FFNN(in_features=2, out_features=2, hidden_size=4, num_layers=2)


@pytest.fixture
def linear_net() -> LinearNetwork:
    """Simple linear network."""
    return LinearNetwork(in_features=2, out_features=2)


class TestResolveHiddenSize:
    def test_returns_explicit_value_when_provided(self) -> None:
        assert _resolve_hidden_size(8, 4, 4) == 8

    def test_defaults_to_in_features_when_square(self) -> None:
        assert _resolve_hidden_size(None, 4, 4) == 4

    def test_raises_when_none_and_not_square(self) -> None:
        with pytest.raises(ValueError, match="hidden_size must be provided"):
            _resolve_hidden_size(None, 4, 6)


class TestFFNNOptionalHiddenSize:
    def test_omit_hidden_size_when_square(self, dense_input: torch.Tensor) -> None:
        m = FFNN(in_features=2, out_features=2, num_layers=2)
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_hidden_size_defaults_to_in_features(self) -> None:
        m = FFNN(in_features=2, out_features=2, num_layers=2)
        assert m.embedding_layer.out_features == 2

    def test_explicit_hidden_size_still_works(self, dense_input: torch.Tensor) -> None:
        m = FFNN(in_features=2, out_features=2, hidden_size=8, num_layers=2)
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_defaults_to_max_when_not_square_and_no_hidden_size(self) -> None:
        m = FFNN(in_features=2, out_features=4, num_layers=2)
        assert m.embedding_layer.out_features == 4


class TestFFNNSkipFalse:
    def test_omit_hidden_size_when_square(self, dense_input: torch.Tensor) -> None:
        m = FFNN(in_features=2, out_features=2, num_layers=2, skip=False)
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_hidden_size_defaults_to_in_features(self) -> None:
        m = FFNN(in_features=2, out_features=2, num_layers=2, skip=False)
        assert m.embedding_layer.out_features == 2

    def test_explicit_hidden_size_still_works(self, dense_input: torch.Tensor) -> None:
        m = FFNN(in_features=2, out_features=2, hidden_size=8, num_layers=2, skip=False)
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_defaults_to_max_when_not_square_and_no_hidden_size(self) -> None:
        m = FFNN(in_features=2, out_features=4, num_layers=2, skip=False)
        assert m.embedding_layer.out_features == 4


class TestVarWidthFFNN:
    """Tests for VarWidthFFNN."""

    def test_output_shape(self, ffnn: VarWidthFFNN, dense_input: torch.Tensor) -> None:
        """Output should have shape (batch, out_features)."""
        assert ffnn(dense_input).shape == (dense_input.shape[0], 2)

    def test_is_dlkit_model(self, ffnn: VarWidthFFNN) -> None:
        """VarWidthFFNN should be instance of nn.Module."""
        assert isinstance(ffnn, nn.Module)

    def test_is_nn_module(self, ffnn: VarWidthFFNN) -> None:
        """VarWidthFFNN should be instance of nn.Module."""
        assert isinstance(ffnn, nn.Module)

    def test_single_hidden_layer(self, dense_input: torch.Tensor) -> None:
        """FFNN with single hidden layer should work."""
        m = VarWidthFFNN(in_features=2, out_features=2, layers=[4])
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_many_hidden_layers(self, dense_input: torch.Tensor) -> None:
        """FFNN with many layers should work."""
        m = VarWidthFFNN(in_features=2, out_features=2, layers=[4, 8, 4, 2])
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_with_batch_norm(self, dense_input: torch.Tensor) -> None:
        """FFNN with batch normalization should work."""
        m = VarWidthFFNN(in_features=2, out_features=2, layers=[4, 4], normalize="batch")
        m.eval()
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_with_layer_norm(self, dense_input: torch.Tensor) -> None:
        """FFNN with layer normalization should work."""
        m = VarWidthFFNN(in_features=2, out_features=2, layers=[4, 4], normalize="layer")
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_with_dropout(self, dense_input: torch.Tensor) -> None:
        """FFNN with dropout should work."""
        m = VarWidthFFNN(in_features=2, out_features=2, layers=[4, 4], dropout=0.2)
        m.eval()
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_has_parameters(self, ffnn: VarWidthFFNN) -> None:
        """FFNN should have trainable parameters."""
        assert len(list(ffnn.parameters())) > 0

    def test_has_embedding_layer(self, ffnn: VarWidthFFNN) -> None:
        """FFNN should have an embedding_layer."""
        assert hasattr(ffnn, "embedding_layer")
        assert isinstance(ffnn.embedding_layer, nn.Linear)
        assert ffnn.embedding_layer.in_features == 2
        assert ffnn.embedding_layer.out_features == 4

    def test_has_regression_layer(self, ffnn: VarWidthFFNN) -> None:
        """FFNN should have a regression_layer."""
        assert hasattr(ffnn, "regression_layer")
        assert isinstance(ffnn.regression_layer, nn.Linear)
        assert ffnn.regression_layer.in_features == 4
        assert ffnn.regression_layer.out_features == 2

    def test_has_hidden_layers(self, ffnn: VarWidthFFNN) -> None:
        """FFNN should have layers ModuleList."""
        assert hasattr(ffnn, "layers")
        assert isinstance(ffnn.layers, nn.ModuleList)
        assert len(ffnn.layers) == 1  # [4 → 4]

    def test_num_layers_stored(self, ffnn: VarWidthFFNN) -> None:
        """FFNN should store num_layers."""
        assert ffnn.num_layers == 1  # transitions in [4, 4]

    def test_activation_stored(self, ffnn: VarWidthFFNN) -> None:
        """FFNN should store activation function."""
        assert hasattr(ffnn, "activation")
        assert callable(ffnn.activation)

    def test_different_widths(self, dense_input: torch.Tensor) -> None:
        """FFNN should support varying layer widths."""
        m = VarWidthFFNN(in_features=2, out_features=2, layers=[8, 4, 6])
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_gradient_flow(self, dense_input: torch.Tensor) -> None:
        """Gradients should flow through all layers."""
        ffnn = VarWidthFFNN(in_features=2, out_features=2, layers=[4, 4])
        out = ffnn(dense_input)
        loss = out.sum()
        loss.backward()
        # All parameters should have gradients
        for param in ffnn.parameters():
            assert param.grad is not None


class TestFFNN:
    """Tests for FFNN."""

    def test_output_shape(self, constant_ffnn: FFNN, dense_input: torch.Tensor) -> None:
        """Output should have shape (batch, out_features)."""
        assert constant_ffnn(dense_input).shape == (dense_input.shape[0], 2)

    def test_is_ffnn(self, constant_ffnn: FFNN) -> None:
        """FFNN should be instance of VarWidthFFNN."""
        assert isinstance(constant_ffnn, VarWidthFFNN)

    def test_is_dlkit_model(self, constant_ffnn: FFNN) -> None:
        """FFNN should be instance of nn.Module."""
        assert isinstance(constant_ffnn, nn.Module)

    def test_is_nn_module(self, constant_ffnn: FFNN) -> None:
        """FFNN should be instance of nn.Module."""
        assert isinstance(constant_ffnn, nn.Module)

    def test_all_hidden_layers_same_width(self) -> None:
        """All hidden layers should have width = hidden_size."""
        m = FFNN(in_features=2, out_features=2, hidden_size=8, num_layers=4)
        # Check that layers were created with constant width
        # First layer: 2 → 8, then 8 → 8 × num_layers hidden transitions.
        assert m.embedding_layer.out_features == 8
        assert m.regression_layer.in_features == 8

    def test_zero_layers_is_valid(self, dense_input: torch.Tensor) -> None:
        """Zero hidden transitions should produce the embed-head degenerate case."""
        m = FFNN(in_features=2, out_features=2, hidden_size=4, num_layers=0)
        assert m(dense_input).shape == (dense_input.shape[0], 2)
        assert len(m.layers) == 0

    def test_single_hidden_layer(self, dense_input: torch.Tensor) -> None:
        """FFNN with one hidden transition should work."""
        m = FFNN(in_features=2, out_features=2, hidden_size=4, num_layers=1)
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_negative_layers_raise(self) -> None:
        """Negative hidden-transition counts should raise ValueError."""
        with pytest.raises(ValueError):
            FFNN(in_features=2, out_features=2, hidden_size=4, num_layers=-1)

    def test_many_hidden_layers(self, dense_input: torch.Tensor) -> None:
        """FFNN with many layers should work."""
        m = FFNN(in_features=2, out_features=2, hidden_size=4, num_layers=5)
        assert m(dense_input).shape == (dense_input.shape[0], 2)

    def test_has_parameters(self, constant_ffnn: FFNN) -> None:
        """FFNN should have trainable parameters."""
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


ShapeMapping = dict[str, tuple[int, ...]]


@pytest.fixture
def rect_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    """Non-square (in=4, out=3) feature/target shape mappings."""
    return {"x": (4,)}, {"y": (3,)}


@pytest.fixture
def square_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    """Square (in=4, out=4) feature/target shape mappings."""
    return {"x": (4,)}, {"y": (4,)}


@pytest.fixture
def nonsquare_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    """Mismatched (in=4, out=3) shape mappings for square-constraint errors."""
    return {"x": (4,)}, {"y": (3,)}


@pytest.fixture
def factorized_linear_net() -> FactorizedLinearNetwork:
    """FactorizedLinearNetwork with in=2, out=2."""
    return FactorizedLinearNetwork(in_features=2, out_features=2)


@pytest.fixture
def symmetric_linear_net() -> SymmetricLinearNetwork:
    """SymmetricLinearNetwork with features=2."""
    return SymmetricLinearNetwork(in_features=2, out_features=2)


@pytest.fixture
def spd_linear_net() -> SPDLinearNetwork:
    """SPDLinearNetwork with features=2."""
    return SPDLinearNetwork(in_features=2, out_features=2)


@pytest.fixture
def symmetric_factorized_net() -> SymmetricFactorizedLinearNetwork:
    """SymmetricFactorizedLinearNetwork with features=2."""
    return SymmetricFactorizedLinearNetwork(in_features=2, out_features=2)


@pytest.fixture
def spd_factorized_net() -> SPDFactorizedLinearNetwork:
    """SPDFactorizedLinearNetwork with features=2."""
    return SPDFactorizedLinearNetwork(in_features=2, out_features=2)


class TestFactorizedLinearNetwork:
    """Tests for FactorizedLinearNetwork."""

    def test_output_shape(
        self, factorized_linear_net: FactorizedLinearNetwork, dense_input: torch.Tensor
    ) -> None:
        """Output should have shape (batch, out_features)."""
        assert factorized_linear_net(dense_input).shape == (dense_input.shape[0], 2)

    def test_no_bias(self, dense_input: torch.Tensor) -> None:
        """bias=False should yield no bias parameter."""
        m = FactorizedLinearNetwork(in_features=2, out_features=2, bias=False)
        assert m(dense_input).shape == (dense_input.shape[0], 2)
        assert m.linear.bias is None

    def test_with_bias(self) -> None:
        """bias=True should store a bias Parameter."""
        m = FactorizedLinearNetwork(in_features=2, out_features=2, bias=True)
        assert isinstance(m.linear.bias, nn.Parameter)

    def test_has_factorized_linear_layer(
        self, factorized_linear_net: FactorizedLinearNetwork
    ) -> None:
        """self.linear should be a FactorizedLinear instance."""
        assert isinstance(factorized_linear_net.linear, FactorizedLinear)

    def test_from_entries(
        self, batch_size: int, rect_shapes: tuple[ShapeMapping, ShapeMapping]
    ) -> None:
        """from_entries should wire in_features and out_features from shapes."""
        in_shapes, out_shapes = rect_shapes
        m = FactorizedLinearNetwork.from_entries(in_shapes, out_shapes)
        assert m(torch.randn(batch_size, in_shapes["x"][0])).shape == (
            batch_size,
            out_shapes["y"][0],
        )

    def test_from_entries_respects_kwargs(
        self, rect_shapes: tuple[ShapeMapping, ShapeMapping]
    ) -> None:
        """from_entries should forward extra kwargs to __init__."""
        in_shapes, out_shapes = rect_shapes
        m = FactorizedLinearNetwork.from_entries(in_shapes, out_shapes, bias=False)
        assert m.linear.bias is None

    def test_gradient_flow(self, dense_input: torch.Tensor) -> None:
        """Gradients should reach base_weight and log_scale."""
        m = FactorizedLinearNetwork(in_features=2, out_features=2)
        m(dense_input).sum().backward()
        assert m.linear.base_weight.grad is not None
        assert m.linear.log_scale.grad is not None

    def test_different_in_out_features(self, batch_size: int) -> None:
        """Asymmetric in/out sizes should be supported."""
        m = FactorizedLinearNetwork(in_features=8, out_features=4)
        assert m(torch.randn(batch_size, 8)).shape == (batch_size, 4)


class TestSymmetricLinearNetwork:
    """Tests for SymmetricLinearNetwork."""

    def test_output_shape(
        self, symmetric_linear_net: SymmetricLinearNetwork, dense_input: torch.Tensor
    ) -> None:
        """Output should have shape (batch, features)."""
        assert symmetric_linear_net(dense_input).shape == (dense_input.shape[0], 2)

    def test_has_symmetric_linear_layer(self, symmetric_linear_net: SymmetricLinearNetwork) -> None:
        """self.linear should be a SymmetricLinear instance."""
        assert isinstance(symmetric_linear_net.linear, SymmetricLinear)

    def test_raises_on_non_square(self) -> None:
        """Non-square shapes should raise ValueError."""
        with pytest.raises(ValueError, match="in_features == out_features"):
            SymmetricLinearNetwork(in_features=3, out_features=2)

    def test_from_entries(
        self, batch_size: int, square_shapes: tuple[ShapeMapping, ShapeMapping]
    ) -> None:
        """from_entries should wire features from square shapes."""
        in_shapes, out_shapes = square_shapes
        m = SymmetricLinearNetwork.from_entries(in_shapes, out_shapes)
        assert m(torch.randn(batch_size, in_shapes["x"][0])).shape == (
            batch_size,
            out_shapes["y"][0],
        )

    def test_from_entries_raises_on_non_square(
        self, nonsquare_shapes: tuple[ShapeMapping, ShapeMapping]
    ) -> None:
        """from_entries with mismatched dimensions should raise ValueError."""
        in_shapes, out_shapes = nonsquare_shapes
        with pytest.raises(ValueError, match="square contract"):
            SymmetricLinearNetwork.from_entries(in_shapes, out_shapes)

    def test_from_entries_respects_kwargs(
        self, square_shapes: tuple[ShapeMapping, ShapeMapping]
    ) -> None:
        """from_entries should forward extra kwargs."""
        in_shapes, out_shapes = square_shapes
        m = SymmetricLinearNetwork.from_entries(in_shapes, out_shapes, bias=True)
        assert m.linear.bias is not None


class TestSPDLinearNetwork:
    """Tests for SPDLinearNetwork."""

    def test_output_shape(
        self, spd_linear_net: SPDLinearNetwork, dense_input: torch.Tensor
    ) -> None:
        """Output should have shape (batch, features)."""
        assert spd_linear_net(dense_input).shape == (dense_input.shape[0], 2)

    def test_has_spd_linear_layer(self, spd_linear_net: SPDLinearNetwork) -> None:
        """self.linear should be a SPDLinear instance."""
        assert isinstance(spd_linear_net.linear, SPDLinear)

    def test_raises_on_non_square(self) -> None:
        """Non-square shapes should raise ValueError."""
        with pytest.raises(ValueError, match="in_features == out_features"):
            SPDLinearNetwork(in_features=3, out_features=2)

    def test_from_entries(
        self, batch_size: int, square_shapes: tuple[ShapeMapping, ShapeMapping]
    ) -> None:
        """from_entries should wire features from square shapes."""
        in_shapes, out_shapes = square_shapes
        m = SPDLinearNetwork.from_entries(in_shapes, out_shapes)
        assert m(torch.randn(batch_size, in_shapes["x"][0])).shape == (
            batch_size,
            out_shapes["y"][0],
        )

    def test_from_entries_raises_on_non_square(
        self, nonsquare_shapes: tuple[ShapeMapping, ShapeMapping]
    ) -> None:
        """from_entries with mismatched dimensions should raise ValueError."""
        in_shapes, out_shapes = nonsquare_shapes
        with pytest.raises(ValueError, match="square contract"):
            SPDLinearNetwork.from_entries(in_shapes, out_shapes)

    def test_gradient_flow(self, dense_input: torch.Tensor) -> None:
        """Gradients should flow through the SPD layer."""
        m = SPDLinearNetwork(in_features=2, out_features=2)
        m(dense_input).sum().backward()
        assert any(p.grad is not None for p in m.parameters())


class TestSymmetricFactorizedLinearNetwork:
    """Tests for SymmetricFactorizedLinearNetwork."""

    def test_output_shape(
        self,
        symmetric_factorized_net: SymmetricFactorizedLinearNetwork,
        dense_input: torch.Tensor,
    ) -> None:
        """Output should have shape (batch, features)."""
        assert symmetric_factorized_net(dense_input).shape == (dense_input.shape[0], 2)

    def test_has_symmetric_factorized_layer(
        self, symmetric_factorized_net: SymmetricFactorizedLinearNetwork
    ) -> None:
        """self.linear should be a SymmetricFactorizedLinear instance."""
        assert isinstance(symmetric_factorized_net.linear, SymmetricFactorizedLinear)

    def test_raises_on_non_square(self) -> None:
        """Non-square shapes should raise ValueError."""
        with pytest.raises(ValueError, match="in_features == out_features"):
            SymmetricFactorizedLinearNetwork(in_features=3, out_features=2)

    def test_from_entries(
        self, batch_size: int, square_shapes: tuple[ShapeMapping, ShapeMapping]
    ) -> None:
        """from_entries should wire features from square shapes."""
        in_shapes, out_shapes = square_shapes
        m = SymmetricFactorizedLinearNetwork.from_entries(in_shapes, out_shapes)
        assert m(torch.randn(batch_size, in_shapes["x"][0])).shape == (
            batch_size,
            out_shapes["y"][0],
        )

    def test_from_entries_raises_on_non_square(
        self, nonsquare_shapes: tuple[ShapeMapping, ShapeMapping]
    ) -> None:
        """from_entries with mismatched dimensions should raise ValueError."""
        in_shapes, out_shapes = nonsquare_shapes
        with pytest.raises(ValueError, match="square contract"):
            SymmetricFactorizedLinearNetwork.from_entries(in_shapes, out_shapes)

    def test_from_entries_respects_kwargs(
        self, batch_size: int, square_shapes: tuple[ShapeMapping, ShapeMapping]
    ) -> None:
        """from_entries should forward extra kwargs."""
        in_shapes, out_shapes = square_shapes
        m = SymmetricFactorizedLinearNetwork.from_entries(in_shapes, out_shapes, std=0.5)
        assert m(torch.randn(batch_size, in_shapes["x"][0])).shape == (
            batch_size,
            out_shapes["y"][0],
        )


class TestSPDFactorizedLinearNetwork:
    """Tests for SPDFactorizedLinearNetwork."""

    def test_output_shape(
        self,
        spd_factorized_net: SPDFactorizedLinearNetwork,
        dense_input: torch.Tensor,
    ) -> None:
        """Output should have shape (batch, features)."""
        assert spd_factorized_net(dense_input).shape == (dense_input.shape[0], 2)

    def test_has_spd_factorized_layer(self, spd_factorized_net: SPDFactorizedLinearNetwork) -> None:
        """self.linear should be a SPDFactorizedLinear instance."""
        assert isinstance(spd_factorized_net.linear, SPDFactorizedLinear)

    def test_raises_on_non_square(self) -> None:
        """Non-square shapes should raise ValueError."""
        with pytest.raises(ValueError, match="in_features == out_features"):
            SPDFactorizedLinearNetwork(in_features=3, out_features=2)

    def test_from_entries(
        self, batch_size: int, square_shapes: tuple[ShapeMapping, ShapeMapping]
    ) -> None:
        """from_entries should wire features from square shapes."""
        in_shapes, out_shapes = square_shapes
        m = SPDFactorizedLinearNetwork.from_entries(in_shapes, out_shapes)
        assert m(torch.randn(batch_size, in_shapes["x"][0])).shape == (
            batch_size,
            out_shapes["y"][0],
        )

    def test_from_entries_raises_on_non_square(
        self, nonsquare_shapes: tuple[ShapeMapping, ShapeMapping]
    ) -> None:
        """from_entries with mismatched dimensions should raise ValueError."""
        in_shapes, out_shapes = nonsquare_shapes
        with pytest.raises(ValueError, match="square contract"):
            SPDFactorizedLinearNetwork.from_entries(in_shapes, out_shapes)

    def test_from_entries_respects_kwargs(
        self, batch_size: int, square_shapes: tuple[ShapeMapping, ShapeMapping]
    ) -> None:
        """from_entries should forward extra kwargs."""
        in_shapes, out_shapes = square_shapes
        m = SPDFactorizedLinearNetwork.from_entries(in_shapes, out_shapes, mean=0.1, std=0.2)
        assert m(torch.randn(batch_size, in_shapes["x"][0])).shape == (
            batch_size,
            out_shapes["y"][0],
        )

    def test_gradient_flow(self, dense_input: torch.Tensor) -> None:
        """Gradients should flow through the SPD factorized layer."""
        m = SPDFactorizedLinearNetwork(in_features=2, out_features=2)
        m(dense_input).sum().backward()
        assert any(p.grad is not None for p in m.parameters())
