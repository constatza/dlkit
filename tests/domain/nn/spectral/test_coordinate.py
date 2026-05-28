"""Tests for coordinate spectral-bias networks."""

from __future__ import annotations

import math

import pytest
import torch

from dlkit.domain.nn.contracts import TabulaRSpec
from dlkit.domain.nn.ffnn.scale_equivariant import ScaleEquivariantFFNN
from dlkit.domain.nn.spectral.coordinate import (
    FourierFeatureNetwork,
    HashEncodingNetwork,
    ModifiedMLP,
    ScaleEquivariantFourierFeatureNetwork,
    ScaleEquivariantModifiedMLP,
    ScaleEquivariantSiren,
    Siren,
)


def _assert_positive_scale_equivariant(model: torch.nn.Module, x: torch.Tensor) -> None:
    scale = 3.5
    y = model(x)
    scaled_y = model(scale * x)
    torch.testing.assert_close(scaled_y, scale * y, atol=1e-4, rtol=1e-4)


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

    def test_from_contract(self) -> None:
        contract = TabulaRSpec(in_shape=(3,), out_shape=(1,))
        net = FourierFeatureNetwork.from_contract(
            contract, hidden_size=16, num_layers=2, n_frequencies=8
        )
        x = torch.randn(4, contract.in_shape[0])
        assert net(x).shape == (4, contract.out_shape[0])


class TestSiren:
    def test_output_shape(self, coords: torch.Tensor) -> None:
        net = Siren(in_features=3, out_features=2, hidden_size=32, num_layers=4)
        assert net(coords).shape == (8, 2)

    def test_activation_gradient_flows(self) -> None:
        net = Siren(in_features=2, out_features=1, hidden_size=8, num_layers=2)
        x = torch.tensor([[math.pi / 2, 0.0]], requires_grad=True)
        out = net(x)
        out.sum().backward()
        assert x.grad is not None

    def test_first_layer_init(self) -> None:
        net = Siren(in_features=4, out_features=1, hidden_size=8, num_layers=3)
        w = net.first_layer.weight.data
        bound = 1.0 / net.first_layer.in_features
        assert w.abs().max() <= bound + 1e-5

    def test_hidden_layer_init(self) -> None:
        net = Siren(in_features=2, out_features=1, hidden_size=8, num_layers=3)
        bound = math.sqrt(6.0 / net.hidden_layers[0].in_features) / net._omega0
        for layer in net.hidden_layers:
            assert layer.weight.data.abs().max() <= bound + 1e-5

    def test_from_contract(self) -> None:
        contract = TabulaRSpec(in_shape=(2,), out_shape=(1,))
        net = Siren.from_contract(contract, hidden_size=16, num_layers=3)
        x = torch.randn(4, contract.in_shape[0])
        assert net(x).shape == (4, contract.out_shape[0])


class TestHashEncodingNetwork:
    def test_output_shape(self, coords: torch.Tensor) -> None:
        net = HashEncodingNetwork(
            in_features=3,
            out_features=2,
            hidden_size=32,
            num_layers=3,
        )
        assert net(coords).shape == (8, 2)

    def test_is_differentiable(self, coords: torch.Tensor) -> None:
        net = HashEncodingNetwork(
            in_features=3,
            out_features=2,
            hidden_size=16,
            num_layers=2,
        )
        coords = coords.requires_grad_(True)
        out = net(coords)
        out.sum().backward()
        assert coords.grad is not None

    def test_from_contract(self) -> None:
        contract = TabulaRSpec(in_shape=(3,), out_shape=(1,))
        net = HashEncodingNetwork.from_contract(contract, hidden_size=16, num_layers=2)
        x = torch.randn(4, contract.in_shape[0])
        assert net(x).shape == (4, contract.out_shape[0])

    def test_rejects_mismatched_bounds(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            HashEncodingNetwork(
                in_features=3,
                out_features=1,
                hidden_size=16,
                num_layers=2,
                bounds=((-1.0, 1.0),),
            )

    def test_include_input_changes_encoding_width(self) -> None:
        with_input = HashEncodingNetwork(
            in_features=3,
            out_features=1,
            hidden_size=8,
            num_layers=2,
            num_levels=4,
            features_per_level=2,
            include_input=True,
        )
        without_input = HashEncodingNetwork(
            in_features=3,
            out_features=1,
            hidden_size=8,
            num_layers=2,
            num_levels=4,
            features_per_level=2,
            include_input=False,
        )
        assert with_input.encoding.output_dim == 11
        assert without_input.encoding.output_dim == 8


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

    def test_from_contract(self) -> None:
        contract = TabulaRSpec(in_shape=(3,), out_shape=(1,))
        net = ModifiedMLP.from_contract(contract, hidden_size=16, num_layers=3)
        x = torch.randn(4, contract.in_shape[0])
        assert net(x).shape == (4, contract.out_shape[0])


@pytest.mark.parametrize(
    ("model_cls", "kwargs"),
    [
        (
            ScaleEquivariantFourierFeatureNetwork,
            {
                "in_features": 3,
                "out_features": 2,
                "hidden_size": 16,
                "num_layers": 2,
                "n_frequencies": 8,
            },
        ),
        (
            ScaleEquivariantSiren,
            {
                "in_features": 3,
                "out_features": 2,
                "hidden_size": 16,
                "num_layers": 2,
            },
        ),
        (
            ScaleEquivariantModifiedMLP,
            {
                "in_features": 3,
                "out_features": 2,
                "hidden_size": 16,
                "num_layers": 3,
            },
        ),
    ],
)
class TestScaleEquivariantCoordinateNetworks:
    def test_output_shape(
        self,
        model_cls: type[torch.nn.Module],
        kwargs: dict[str, int],
        coords: torch.Tensor,
    ) -> None:
        net = model_cls(**kwargs)
        assert net(coords).shape == (8, 2)

    def test_positive_scale_equivariance(
        self,
        model_cls: type[torch.nn.Module],
        kwargs: dict[str, int],
        coords: torch.Tensor,
    ) -> None:
        torch.manual_seed(0)
        net = model_cls(**kwargs)
        _assert_positive_scale_equivariant(net, coords)

    def test_keep_stats(
        self,
        model_cls: type[torch.nn.Module],
        kwargs: dict[str, int],
        coords: torch.Tensor,
    ) -> None:
        net = model_cls(**kwargs, keep_stats=True)
        output, stats = net(coords)
        assert output.shape == (8, 2)
        assert set(stats) == {"norm"}
        assert stats["norm"].shape == (8, 1)

    def test_integer_input_rejected(
        self,
        model_cls: type[torch.nn.Module],
        kwargs: dict[str, int],
    ) -> None:
        net = model_cls(**kwargs)
        with pytest.raises(TypeError, match="floating point tensor"):
            net(torch.ones((4, kwargs["in_features"]), dtype=torch.int64))

    def test_invalid_norm_rejected(
        self,
        model_cls: type[torch.nn.Module],
        kwargs: dict[str, int],
    ) -> None:
        with pytest.raises(ValueError, match="norm must be one of"):
            model_cls(**kwargs, norm="l0")

    def test_non_positive_eps_gain_rejected(
        self,
        model_cls: type[torch.nn.Module],
        kwargs: dict[str, int],
    ) -> None:
        with pytest.raises(ValueError, match="eps_gain must be > 0"):
            model_cls(**kwargs, eps_gain=0.0)


class TestContractAwareScaleEquivariantCoordinateNetworks:
    def test_fourier_feature_from_contract(self) -> None:
        contract = TabulaRSpec(in_shape=(3,), out_shape=(2,))
        net = ScaleEquivariantFourierFeatureNetwork.from_contract(
            contract,
            hidden_size=16,
            num_layers=2,
            n_frequencies=8,
        )
        assert net(torch.randn(4, contract.in_shape[0])).shape == (4, contract.out_shape[0])

    def test_siren_from_contract(self) -> None:
        contract = TabulaRSpec(in_shape=(3,), out_shape=(2,))
        net = ScaleEquivariantSiren.from_contract(contract, hidden_size=16, num_layers=2)
        assert net(torch.randn(4, contract.in_shape[0])).shape == (4, contract.out_shape[0])

    def test_modified_mlp_from_contract(self) -> None:
        contract = TabulaRSpec(in_shape=(3,), out_shape=(2,))
        net = ScaleEquivariantModifiedMLP.from_contract(contract, hidden_size=16, num_layers=3)
        assert net(torch.randn(4, contract.in_shape[0])).shape == (4, contract.out_shape[0])

    def test_hash_encoding_from_contract(self) -> None:
        contract = TabulaRSpec(in_shape=(3,), out_shape=(2,))
        net = HashEncodingNetwork.from_contract(contract, hidden_size=16, num_layers=2)
        assert net(torch.randn(4, contract.in_shape[0])).shape == (4, contract.out_shape[0])


class TestScaleEquivariantDenseRegression:
    def test_existing_dense_model_remains_positive_scale_equivariant(self) -> None:
        torch.manual_seed(0)
        net = ScaleEquivariantFFNN(
            in_features=3,
            out_features=2,
            hidden_size=16,
            num_layers=2,
        )
        _assert_positive_scale_equivariant(net, torch.randn(8, 3))
