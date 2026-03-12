from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from dlkit.core.models.nn.primitives.parametrized_layers import (
    FactorizedLinear,
    register_spd,
    register_spd_factorized,
    register_symmetric,
    register_symmetric_factorized,
    SPDFactorizedLinear,
    SPDLinear,
    SymmetricFactorizedLinear,
    SymmetricLinear,
)


def _weight(module: nn.Module) -> torch.Tensor:
    """Return the effective weight of *module* as a Tensor.

    PyTorch's parametrize machinery makes ``module.weight`` opaque to static
    type checkers (``Tensor | Module``); this helper encapsulates the cast so
    individual tests remain readable.

    Args:
        module: Module whose ``weight`` attribute is to be retrieved.

    Returns:
        The effective weight tensor.
    """
    return cast(torch.Tensor, module.weight)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def square_linear() -> nn.Linear:
    """Square 4×4 linear layer (no bias)."""
    return nn.Linear(4, 4, bias=False)


@pytest.fixture
def rectangular_linear() -> nn.Linear:
    """Rectangular 3×2 linear layer (no bias)."""
    return nn.Linear(2, 3, bias=False)


@pytest.fixture
def symmetric_linear() -> SymmetricLinear:
    """SymmetricLinear layer of size 4."""
    return SymmetricLinear(4, bias=False)


@pytest.fixture
def spd_linear() -> SPDLinear:
    """SPDLinear layer of size 4."""
    return SPDLinear(4, bias=False, min_diag=1e-4)


@pytest.fixture
def factorized_linear() -> FactorizedLinear:
    """FactorizedLinear layer with in=2, out=3."""
    return FactorizedLinear(2, 3, bias=False)


@pytest.fixture
def symmetric_factorized_linear() -> SymmetricFactorizedLinear:
    """SymmetricFactorizedLinear layer of size 4."""
    return SymmetricFactorizedLinear(4, bias=False)


@pytest.fixture
def spd_factorized_linear() -> SPDFactorizedLinear:
    """SPDFactorizedLinear layer of size 4."""
    return SPDFactorizedLinear(4, bias=False, min_diag=1e-4)


# ---------------------------------------------------------------------------
# Tests — register_symmetric
# ---------------------------------------------------------------------------


class TestRegisterSymmetric:
    """Tests for :func:`register_symmetric`."""

    def test_weight_is_symmetric(self, square_linear: nn.Linear) -> None:
        """After registration the effective weight must be symmetric."""
        layer = register_symmetric(square_linear)
        w = _weight(layer)
        assert torch.allclose(w, w.T)

    def test_parametrization_is_registered(self, square_linear: nn.Linear) -> None:
        """register_symmetric must mark the weight as parametrized."""
        layer = register_symmetric(square_linear)
        assert parametrize.is_parametrized(layer, "weight")

    def test_returns_same_module(self, square_linear: nn.Linear) -> None:
        """register_symmetric must return the same module object."""
        result = register_symmetric(square_linear)
        assert result is square_linear


# ---------------------------------------------------------------------------
# Tests — register_spd
# ---------------------------------------------------------------------------


class TestRegisterSPD:
    """Tests for :func:`register_spd`."""

    def test_weight_is_symmetric(self, square_linear: nn.Linear) -> None:
        """After registration the effective weight must be symmetric."""
        layer = register_spd(square_linear, min_diag=1e-4)
        w = _weight(layer)
        assert torch.allclose(w, w.T, atol=1e-6)

    def test_weight_is_positive_definite(self, square_linear: nn.Linear) -> None:
        """After registration all eigenvalues must be positive."""
        layer = register_spd(square_linear, min_diag=1e-4)
        eigvals = torch.linalg.eigvalsh(_weight(layer))
        assert torch.all(eigvals > 0.0)

    def test_parametrization_is_registered(self, square_linear: nn.Linear) -> None:
        """register_spd must mark the weight as parametrized."""
        layer = register_spd(square_linear)
        assert parametrize.is_parametrized(layer, "weight")


# ---------------------------------------------------------------------------
# Tests — register_symmetric_factorized
# ---------------------------------------------------------------------------


class TestRegisterSymmetricFactorized:
    """Tests for :func:`register_symmetric_factorized`."""

    def test_weight_is_symmetric(self, square_linear: nn.Linear) -> None:
        """After registration the effective weight must be symmetric."""
        layer = register_symmetric_factorized(square_linear, size=4)
        w = _weight(layer)
        assert torch.allclose(w, w.T)

    def test_parametrization_is_registered(self, square_linear: nn.Linear) -> None:
        """register_symmetric_factorized must mark the weight as parametrized."""
        layer = register_symmetric_factorized(square_linear, size=4)
        assert parametrize.is_parametrized(layer, "weight")


# ---------------------------------------------------------------------------
# Tests — register_spd_factorized
# ---------------------------------------------------------------------------


class TestRegisterSPDFactorized:
    """Tests for :func:`register_spd_factorized`."""

    def test_weight_is_symmetric(self, square_linear: nn.Linear) -> None:
        """After registration the effective weight must be symmetric."""
        layer = register_spd_factorized(square_linear, size=4, min_diag=1e-4)
        w = _weight(layer)
        assert torch.allclose(w, w.T, atol=1e-6)

    def test_weight_is_positive_definite(self, square_linear: nn.Linear) -> None:
        """After registration all eigenvalues must be positive."""
        layer = register_spd_factorized(square_linear, size=4, min_diag=1e-4)
        eigvals = torch.linalg.eigvalsh(_weight(layer))
        assert torch.all(eigvals > 0.0)


# ---------------------------------------------------------------------------
# Tests — SymmetricLinear
# ---------------------------------------------------------------------------


class TestSymmetricLinear:
    """Tests for :class:`SymmetricLinear`."""

    def test_weight_is_symmetric_at_construction(
        self,
        symmetric_linear: SymmetricLinear,
    ) -> None:
        """SymmetricLinear must expose a symmetric weight immediately."""
        w = _weight(symmetric_linear)
        assert torch.allclose(w, w.T)

    def test_parametrization_is_registered(
        self,
        symmetric_linear: SymmetricLinear,
    ) -> None:
        """SymmetricLinear must register the weight parametrization."""
        assert parametrize.is_parametrized(symmetric_linear, "weight")

    def test_in_equals_out_features(self, symmetric_linear: SymmetricLinear) -> None:
        """Input and output features must be equal."""
        assert symmetric_linear.in_features == symmetric_linear.out_features

    def test_forward_produces_correct_output_shape(
        self,
        symmetric_linear: SymmetricLinear,
    ) -> None:
        """forward() must return (batch, features) given (batch, features) input."""
        x = torch.randn(5, 4)
        y = symmetric_linear(x)
        assert y.shape == (5, 4)

    def test_gradient_flows(self, symmetric_linear: SymmetricLinear) -> None:
        """Gradients must flow through the symmetric weight."""
        x = torch.randn(2, 4)
        symmetric_linear(x).sum().backward()
        assert any(p.grad is not None for p in symmetric_linear.parameters())


# ---------------------------------------------------------------------------
# Tests — SPDLinear
# ---------------------------------------------------------------------------


class TestSPDLinear:
    """Tests for :class:`SPDLinear`."""

    def test_weight_is_spd_at_construction(self, spd_linear: SPDLinear) -> None:
        """SPDLinear must expose an SPD weight immediately."""
        w = _weight(spd_linear)
        assert torch.allclose(w, w.T, atol=1e-6)
        eigvals = torch.linalg.eigvalsh(w)
        assert torch.all(eigvals > 0.0)

    def test_parametrization_is_registered(self, spd_linear: SPDLinear) -> None:
        """SPDLinear must register the weight parametrization."""
        assert parametrize.is_parametrized(spd_linear, "weight")

    def test_forward_produces_correct_output_shape(
        self,
        spd_linear: SPDLinear,
    ) -> None:
        """forward() must return (batch, features) given (batch, features) input."""
        x = torch.randn(5, 4)
        y = spd_linear(x)
        assert y.shape == (5, 4)

    def test_gradient_flows(self, spd_linear: SPDLinear) -> None:
        """Gradients must flow through the SPD weight."""
        x = torch.randn(2, 4)
        spd_linear(x).sum().backward()
        assert any(p.grad is not None for p in spd_linear.parameters())


# ---------------------------------------------------------------------------
# Tests — FactorizedLinear
# ---------------------------------------------------------------------------


class TestFactorizedLinear:
    """Tests for :class:`FactorizedLinear`."""

    def test_weight_property_has_correct_shape(
        self,
        factorized_linear: FactorizedLinear,
    ) -> None:
        """weight property must return a (out_features, in_features) tensor."""
        assert factorized_linear.weight.shape == (3, 2)

    def test_base_weight_has_correct_shape(
        self,
        factorized_linear: FactorizedLinear,
    ) -> None:
        """base_weight must be (out_features, in_features)."""
        assert factorized_linear.base_weight.shape == (3, 2)

    def test_log_scale_has_correct_shape(
        self,
        factorized_linear: FactorizedLinear,
    ) -> None:
        """log_scale must be (out_features,)."""
        assert factorized_linear.log_scale.shape == (3,)

    def test_no_bias_when_disabled(self) -> None:
        """bias must be None when bias=False."""
        layer = FactorizedLinear(2, 3, bias=False)
        assert layer.bias is None

    def test_bias_exists_when_enabled(self) -> None:
        """bias must be a parameter when bias=True."""
        layer = FactorizedLinear(2, 3, bias=True)
        assert isinstance(layer.bias, nn.Parameter)
        assert layer.bias.shape == (3,)

    def test_forward_produces_correct_output_shape(
        self,
        factorized_linear: FactorizedLinear,
    ) -> None:
        """forward() must return (batch, out_features) given (batch, in_features)."""
        x = torch.randn(5, 2)
        y = factorized_linear(x)
        assert y.shape == (5, 3)

    def test_gradient_flows_through_scale_and_base(
        self,
        factorized_linear: FactorizedLinear,
    ) -> None:
        """Gradients must reach both base_weight and log_scale."""
        x = torch.randn(2, 2)
        factorized_linear(x).sum().backward()
        assert factorized_linear.base_weight.grad is not None
        assert factorized_linear.log_scale.grad is not None

    def test_weight_reflects_scale_change(self) -> None:
        """Setting log_scale to 0 must yield effective weight equal to base_weight."""
        layer = FactorizedLinear(2, 3, bias=False)
        with torch.no_grad():
            layer.log_scale.fill_(0.0)
        assert torch.allclose(layer.weight, layer.base_weight)

    def test_not_using_parametrize_machinery(
        self,
        factorized_linear: FactorizedLinear,
    ) -> None:
        """FactorizedLinear must NOT use parametrize (flat state dict)."""
        assert not parametrize.is_parametrized(factorized_linear, "base_weight")

    def test_default_mean_gives_near_unit_scale(self) -> None:
        """Default mean=0.0 should give log_scale ≈ 0."""
        layer = FactorizedLinear(8, 16)
        assert layer.log_scale.abs().mean().item() < 0.5


# ---------------------------------------------------------------------------
# Tests — SymmetricFactorizedLinear
# ---------------------------------------------------------------------------


class TestSymmetricFactorizedLinear:
    """Tests for :class:`SymmetricFactorizedLinear`."""

    def test_weight_is_symmetric_at_construction(
        self,
        symmetric_factorized_linear: SymmetricFactorizedLinear,
    ) -> None:
        """Weight must be symmetric immediately after construction."""
        w = _weight(symmetric_factorized_linear)
        assert torch.allclose(w, w.T)

    def test_parametrization_is_registered(
        self,
        symmetric_factorized_linear: SymmetricFactorizedLinear,
    ) -> None:
        """Must register weight parametrization."""
        assert parametrize.is_parametrized(symmetric_factorized_linear, "weight")

    def test_forward_produces_correct_output_shape(
        self,
        symmetric_factorized_linear: SymmetricFactorizedLinear,
    ) -> None:
        """forward() must return (batch, features) output."""
        x = torch.randn(5, 4)
        y = symmetric_factorized_linear(x)
        assert y.shape == (5, 4)

    def test_gradient_flows(
        self,
        symmetric_factorized_linear: SymmetricFactorizedLinear,
    ) -> None:
        """Gradients must propagate through the factorized symmetric weight."""
        x = torch.randn(2, 4)
        symmetric_factorized_linear(x).sum().backward()
        assert any(
            p.grad is not None for p in symmetric_factorized_linear.parameters()
        )


# ---------------------------------------------------------------------------
# Tests — SPDFactorizedLinear
# ---------------------------------------------------------------------------


class TestSPDFactorizedLinear:
    """Tests for :class:`SPDFactorizedLinear`."""

    def test_weight_is_spd_at_construction(
        self,
        spd_factorized_linear: SPDFactorizedLinear,
    ) -> None:
        """Weight must be SPD immediately after construction."""
        w = _weight(spd_factorized_linear)
        assert torch.allclose(w, w.T, atol=1e-6)
        eigvals = torch.linalg.eigvalsh(w)
        assert torch.all(eigvals > 0.0)

    def test_parametrization_is_registered(
        self,
        spd_factorized_linear: SPDFactorizedLinear,
    ) -> None:
        """Must register weight parametrization."""
        assert parametrize.is_parametrized(spd_factorized_linear, "weight")

    def test_forward_produces_correct_output_shape(
        self,
        spd_factorized_linear: SPDFactorizedLinear,
    ) -> None:
        """forward() must return (batch, features) output."""
        x = torch.randn(5, 4)
        y = spd_factorized_linear(x)
        assert y.shape == (5, 4)

    def test_gradient_flows(
        self,
        spd_factorized_linear: SPDFactorizedLinear,
    ) -> None:
        """Gradients must propagate through the factorized SPD weight."""
        x = torch.randn(2, 4)
        spd_factorized_linear(x).sum().backward()
        assert any(p.grad is not None for p in spd_factorized_linear.parameters())
