"""Tests for activation-driven weight initialization (``domain.nn.init``)."""

from __future__ import annotations

from unittest.mock import patch

import torch
from torch import nn

from dlkit.domain.nn.init import initialize_, resolve_initializer


def test_relu_family_uses_kaiming_with_relu_gain() -> None:
    """relu/gelu/silu/none all resolve to Kaiming with the ReLU nonlinearity gain."""
    weight = torch.empty(4, 4)
    for name in ("relu", "gelu", "silu", "none", "identity"):
        with patch("torch.nn.init.kaiming_uniform_") as mock_kaiming:
            resolve_initializer(name)(weight)
        _, kwargs = mock_kaiming.call_args
        assert kwargs["nonlinearity"] == "relu"


def test_leaky_relu_uses_kaiming_with_negative_slope() -> None:
    """leaky_relu passes its negative slope through to kaiming_uniform_."""
    weight = torch.empty(4, 4)
    with patch("torch.nn.init.kaiming_uniform_") as mock_kaiming:
        resolve_initializer("leaky_relu")(weight)
    _, kwargs = mock_kaiming.call_args
    assert kwargs["a"] == 0.01
    assert kwargs["nonlinearity"] == "leaky_relu"


def test_symmetric_activations_use_xavier_with_matching_gain() -> None:
    """tanh/sigmoid resolve to Xavier scaled by their own calculate_gain."""
    weight = torch.empty(4, 4)
    for name in ("tanh", "sigmoid"):
        with patch("torch.nn.init.xavier_uniform_") as mock_xavier:
            resolve_initializer(name)(weight)
        _, kwargs = mock_xavier.call_args
        assert kwargs["gain"] == nn.init.calculate_gain(name)


def test_unsupported_activation_raises() -> None:
    """An unrecognised activation name fails loudly instead of silently defaulting."""
    try:
        resolve_initializer("not-a-real-activation")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unsupported activation")


def test_initialize_only_touches_linear_and_conv_weights() -> None:
    """initialize_ walks the module tree, initializing only Linear/Conv weights and zeroing bias."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.LayerNorm(4))
    before = model[0].weight.clone()

    initialize_(model, "relu")

    assert not torch.equal(before, model[0].weight)
    assert torch.equal(model[0].bias, torch.zeros_like(model[0].bias))
