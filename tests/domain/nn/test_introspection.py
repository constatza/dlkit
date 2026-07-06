"""Tests for effective (post-default-resolution) hyperparameter recovery.

Covers HyperparameterAware protocol matching and the merge semantics of
effective_hyperparameters against VarWidthFFNN, FFNN, and FourierNeuralOperator1d.
"""

from __future__ import annotations

import pytest
from torch import nn

from dlkit.domain.nn.contracts import HyperparameterAware
from dlkit.domain.nn.ffnn.residual import FFNN, VarWidthFFNN
from dlkit.domain.nn.introspection import effective_hyperparameters
from dlkit.domain.nn.operators.fno import FourierNeuralOperator1d


@pytest.fixture
def var_width_ffnn() -> VarWidthFFNN:
    """VarWidthFFNN built with activation left unset (resolves to relu)."""
    return VarWidthFFNN(in_features=2, out_features=2, layers=[4, 4])


@pytest.fixture
def ffnn() -> FFNN:
    """FFNN built with activation left unset (resolves to gelu)."""
    return FFNN(in_features=2, out_features=2, hidden_size=4, num_layers=2)


@pytest.fixture
def fno() -> FourierNeuralOperator1d:
    """FourierNeuralOperator1d built with activation left unset (resolves to relu)."""
    return FourierNeuralOperator1d(in_channels=2, out_channels=2, n_modes=8)


class TestHyperparameterAwareProtocol:
    def test_model_with_hyperparameters_dict_matches(self, var_width_ffnn: VarWidthFFNN) -> None:
        assert isinstance(var_width_ffnn, HyperparameterAware)

    def test_plain_module_does_not_match(self) -> None:
        assert not isinstance(nn.Linear(2, 2), HyperparameterAware)


class TestEffectiveHyperparameters:
    def test_var_width_ffnn_resolves_activation_default(self, var_width_ffnn: VarWidthFFNN) -> None:
        result = effective_hyperparameters(var_width_ffnn, overrides={})
        assert result["activation"] == "relu"
        assert result["num_layers"] == 1

    def test_ffnn_resolves_gelu_default(self, ffnn: FFNN) -> None:
        result = effective_hyperparameters(ffnn, overrides={})
        assert result["activation"] == "gelu"

    def test_fno_resolves_activation_default(self, fno: FourierNeuralOperator1d) -> None:
        result = effective_hyperparameters(fno, overrides={})
        assert result["activation"] == "relu"
        assert result["n_modes"] == 8

    def test_declared_values_win_over_overrides(self, var_width_ffnn: VarWidthFFNN) -> None:
        result = effective_hyperparameters(
            var_width_ffnn, overrides={"activation": "stale-override-value"}
        )
        assert result["activation"] == "relu"

    def test_overrides_survive_when_not_declared(self, var_width_ffnn: VarWidthFFNN) -> None:
        result = effective_hyperparameters(
            var_width_ffnn, overrides={"in_features": 2, "out_features": 2}
        )
        assert result["in_features"] == 2
        assert result["out_features"] == 2

    def test_plain_module_degrades_to_overrides_only(self) -> None:
        model = nn.Linear(2, 2)
        result = effective_hyperparameters(model, overrides={"hidden_size": 4})
        assert result == {"hidden_size": 4}
