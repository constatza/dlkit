"""Tests for _build_model_from_settings: normalize forwarding and effective hparams.

Regression coverage for two fixes:
- `normalize` is forwarded to the model constructor like `activation` already was
  (previously silently dropped even when explicitly set).
- The returned effective-hyperparameters dict reflects what the model actually
  resolved (e.g. activation defaults), not just the raw settings.
"""

from __future__ import annotations

from torch import nn

from dlkit.engine.adapters.lightning.base import _build_model_from_settings
from dlkit.infrastructure.config.model_components import ModelComponentSettings


class TestNormalizeForwarding:
    def test_explicit_normalize_is_applied(self) -> None:
        settings = ModelComponentSettings(
            name="FFNN",
            module_path="dlkit.domain.nn",
            in_features=2,
            out_features=2,
            hidden_size=4,
            num_layers=1,
            normalize="batch",
        )
        model, _ = _build_model_from_settings(settings)
        norm_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
        assert norm_layers, "expected explicit normalize='batch' to produce BatchNorm1d layers"

    def test_unset_normalize_stays_identity(self) -> None:
        settings = ModelComponentSettings(
            name="FFNN",
            module_path="dlkit.domain.nn",
            in_features=2,
            out_features=2,
            hidden_size=4,
            num_layers=1,
        )
        model, _ = _build_model_from_settings(settings)
        norm_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
        assert not norm_layers


class TestEffectiveHyperparametersReturned:
    def test_resolved_activation_reflects_model_default(self) -> None:
        settings = ModelComponentSettings(
            name="FFNN",
            module_path="dlkit.domain.nn",
            in_features=2,
            out_features=2,
            hidden_size=4,
            num_layers=1,
            activation=None,
        )
        _, hyperparameters = _build_model_from_settings(settings)
        assert hyperparameters["activation"] == "gelu"

    def test_forwarded_settings_survive_in_hyperparameters(self) -> None:
        settings = ModelComponentSettings(
            name="FFNN",
            module_path="dlkit.domain.nn",
            in_features=2,
            out_features=2,
            hidden_size=4,
            num_layers=1,
        )
        _, hyperparameters = _build_model_from_settings(settings)
        assert hyperparameters["hidden_size"] == 4
        assert hyperparameters["num_layers"] == 1
