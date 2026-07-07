"""Tests for the row-wise factorized linear primitives.

Kaiming gain exposure
---------------------
``SoftplusFactorizedLinear`` exposes ``kaiming_a`` (the ``a`` gain passed to
``nn.init.kaiming_uniform_`` on ``base_weight``) as a constructor parameter,
defaulting to ``0.0`` (the textbook He/Kaiming gain for ReLU/GELU-family
activations), so callers can override it explicitly instead of it being
hardcoded inside the primitive.
"""

from __future__ import annotations

from unittest.mock import patch

from dlkit.domain.nn.primitives import SoftplusFactorizedLinear


def test_softplus_factorized_linear_default_kaiming_a() -> None:
    """Default kaiming_a is 0.0 (He/Kaiming gain for ReLU/GELU-family activations)."""
    with patch("torch.nn.init.kaiming_uniform_") as mock_kaiming_uniform:
        SoftplusFactorizedLinear(in_features=4, out_features=4)

    _, kwargs = mock_kaiming_uniform.call_args
    assert kwargs["a"] == 0.0


def test_softplus_factorized_linear_exposes_kaiming_a() -> None:
    """kaiming_a is forwarded to kaiming_uniform_ when overridden explicitly."""
    custom_kaiming_a = 5**0.5
    with patch("torch.nn.init.kaiming_uniform_") as mock_kaiming_uniform:
        SoftplusFactorizedLinear(in_features=4, out_features=4, kaiming_a=custom_kaiming_a)

    _, kwargs = mock_kaiming_uniform.call_args
    assert kwargs["a"] == custom_kaiming_a
