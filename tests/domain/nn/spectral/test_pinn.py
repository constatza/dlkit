"""Tests for PINN-oriented frequency and coordinate encoding networks."""

from __future__ import annotations

import math

import torch

from dlkit.domain.nn.spectral.pinn import FourierFeatureNetwork, ModifiedMLP, SirenFFNN


class TestFourierFeatureNetwork:
    def test_output_shape(self, coords: torch.Tensor) -> None:
        net = FourierFeatureNetwork(
            in_features=3, out_features=2, hidden_size=32, num_layers=3, n_frequencies=16
        )
        assert net(coords).shape == (8, 2)

    def test_fixed_b_is_not_param(self) -> None:
        net = FourierFeatureNetwork(
            in_features=3,
            out_features=2,
            hidden_size=32,
            num_layers=3,
            n_frequencies=16,
            learnable_B=False,
        )
        param_names = {n for n, _ in net.named_parameters()}
        assert not any("B" in n for n in param_names)

    def test_learnable_b_is_param(self) -> None:
        net = FourierFeatureNetwork(
            in_features=3,
            out_features=2,
            hidden_size=32,
            num_layers=3,
            n_frequencies=16,
            learnable_B=True,
        )
        param_names = {n for n, _ in net.named_parameters()}
        assert any("B" in n for n in param_names)

    def test_from_shape(self) -> None:
        from dlkit.common.shapes import ShapeSummary

        shape = ShapeSummary(in_shapes=((3,),), out_shapes=((1,),))
        net = FourierFeatureNetwork.from_shape(shape, hidden_size=16, num_layers=2, n_frequencies=8)
        x = torch.randn(4, 3)
        assert net(x).shape == (4, 1)


class TestSirenFFNN:
    def test_output_shape(self, coords: torch.Tensor) -> None:
        net = SirenFFNN(in_features=3, out_features=2, hidden_size=32, num_layers=4)
        assert net(coords).shape == (8, 2)

    def test_activation_gradient_flows(self) -> None:
        net = SirenFFNN(in_features=2, out_features=1, hidden_size=8, num_layers=2)
        x = torch.tensor([[math.pi / 2, 0.0]], requires_grad=True)
        out = net(x)
        out.sum().backward()
        assert x.grad is not None

    def test_first_layer_init(self) -> None:
        net = SirenFFNN(in_features=4, out_features=1, hidden_size=8, num_layers=3)
        w = net.first_layer.weight.data
        bound = 1.0 / net.first_layer.in_features
        assert w.abs().max() <= bound + 1e-5

    def test_hidden_layer_init(self) -> None:
        net = SirenFFNN(in_features=2, out_features=1, hidden_size=8, num_layers=3)
        bound = math.sqrt(6.0 / net.hidden_layers[0].in_features) / net._omega0
        for layer in net.hidden_layers:
            assert layer.weight.data.abs().max() <= bound + 1e-5

    def test_from_shape(self) -> None:
        from dlkit.common.shapes import ShapeSummary

        shape = ShapeSummary(in_shapes=((2,),), out_shapes=((1,),))
        net = SirenFFNN.from_shape(shape, hidden_size=16, num_layers=3)
        x = torch.randn(4, 2)
        assert net(x).shape == (4, 1)


class TestModifiedMLP:
    def test_output_shape(self, coords: torch.Tensor) -> None:
        net = ModifiedMLP(in_features=3, out_features=2, hidden_size=32, num_layers=4)
        assert net(coords).shape == (8, 2)

    def test_gradient_flows(self, coords: torch.Tensor) -> None:
        net = ModifiedMLP(in_features=3, out_features=2, hidden_size=16, num_layers=3)
        coords = coords.requires_grad_(True)
        out = net(coords)
        out.sum().backward()
        assert coords.grad is not None

    def test_from_shape(self) -> None:
        from dlkit.common.shapes import ShapeSummary

        shape = ShapeSummary(in_shapes=((3,),), out_shapes=((1,),))
        net = ModifiedMLP.from_shape(shape, hidden_size=16, num_layers=3)
        x = torch.randn(4, 3)
        assert net(x).shape == (4, 1)
