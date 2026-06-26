"""Tests for GatedMLP feed-forward network with pluggable gating.

All input tensors are provided via fixtures.  No data is created inside
test functions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
import torch
from torch import nn

from dlkit.common.shapes import ShapeContext
from dlkit.domain.nn.contracts import HyperParam
from dlkit.domain.nn.ffnn.gated import GatedMLP

ShapeMapping = dict[str, tuple[int, ...]]

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_IN = 4
_OUT = 2
_HIDDEN = 8
_LAYERS = 2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGatedMLP:
    """Tests for GatedMLP."""

    def test_output_shape_glu(
        self,
        glu_factory: Callable,
        network_input: torch.Tensor,
    ) -> None:
        """GatedMLP with GLUGate produces correct output shape."""
        model = GatedMLP(
            in_features=_IN,
            out_features=_OUT,
            hidden_size=_HIDDEN,
            num_layers=_LAYERS,
            gate_factory=glu_factory,
        )
        out = model(network_input)
        assert out.shape == (network_input.shape[0], _OUT)

    def test_output_shape_swiglu(
        self,
        swiglu_factory: Callable,
        network_input: torch.Tensor,
    ) -> None:
        """GatedMLP with SwiGLUGate produces correct output shape."""
        model = GatedMLP(
            in_features=_IN,
            out_features=_OUT,
            hidden_size=_HIDDEN,
            num_layers=_LAYERS,
            gate_factory=swiglu_factory,
        )
        out = model(network_input)
        assert out.shape == (network_input.shape[0], _OUT)

    def test_output_shape_grn(
        self,
        grn_compatible_factory: Callable,
        network_input: torch.Tensor,
    ) -> None:
        """GatedMLP with GRNGate(context_size=4) produces correct output shape."""
        model = GatedMLP(
            in_features=_IN,
            out_features=_OUT,
            hidden_size=_HIDDEN,
            num_layers=1,
            gate_factory=grn_compatible_factory,
        )
        out = model(network_input)
        assert out.shape == (network_input.shape[0], _OUT)

    def test_output_shape_uv(
        self,
        uv_factory: Callable,
        network_input: torch.Tensor,
    ) -> None:
        """GatedMLP with UVGate produces correct output shape."""
        model = GatedMLP(
            in_features=_IN,
            out_features=_OUT,
            hidden_size=_HIDDEN,
            num_layers=_LAYERS,
            gate_factory=uv_factory,
        )
        out = model(network_input)
        assert out.shape == (network_input.shape[0], _OUT)

    def test_zero_layers_raises(self, glu_factory: Callable) -> None:
        """num_layers=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_layers"):
            GatedMLP(
                in_features=_IN,
                out_features=_OUT,
                hidden_size=_HIDDEN,
                num_layers=0,
                gate_factory=glu_factory,
            )

    def test_gradient_flows(
        self,
        glu_factory: Callable,
        network_input: torch.Tensor,
    ) -> None:
        """Backward pass populates gradients on all parameters."""
        model = GatedMLP(
            in_features=_IN,
            out_features=_OUT,
            hidden_size=_HIDDEN,
            num_layers=_LAYERS,
            gate_factory=glu_factory,
        )
        out = model(network_input)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_with_normalization(
        self,
        glu_factory: Callable,
        network_input: torch.Tensor,
    ) -> None:
        """GatedMLP with layer normalisation runs without error."""
        model = GatedMLP(
            in_features=_IN,
            out_features=_OUT,
            hidden_size=_HIDDEN,
            num_layers=_LAYERS,
            gate_factory=glu_factory,
            normalize="layer",
        )
        out = model(network_input)
        assert out.shape == (network_input.shape[0], _OUT)

    def test_with_dropout(
        self,
        glu_factory: Callable,
        network_input: torch.Tensor,
    ) -> None:
        """GatedMLP with dropout=0.1 runs in eval mode without error."""
        model = GatedMLP(
            in_features=_IN,
            out_features=_OUT,
            hidden_size=_HIDDEN,
            num_layers=_LAYERS,
            gate_factory=glu_factory,
            dropout=0.1,
        )
        model.eval()
        out = model(network_input)
        assert out.shape == (network_input.shape[0], _OUT)

    def test_from_entries(
        self,
        glu_factory: Callable,
        tabular_shapes: tuple[ShapeMapping, ShapeMapping],
        network_input: torch.Tensor,
    ) -> None:
        """from_entries classmethod constructs a working GatedMLP."""
        in_shapes, out_shapes = tabular_shapes
        model = GatedMLP.from_context(
            ShapeContext(in_shapes, out_shapes),
            hidden_size=_HIDDEN,
            num_layers=_LAYERS,
            gate_factory=cast(HyperParam, glu_factory),
        )
        out = model(network_input)
        assert out.shape == (network_input.shape[0], _OUT)

    def test_gates_are_independent(self, glu_factory: Callable) -> None:
        """Each layer has its own gate instance (no shared parameters)."""
        model = GatedMLP(
            in_features=_IN,
            out_features=_OUT,
            hidden_size=_HIDDEN,
            num_layers=_LAYERS,
            gate_factory=glu_factory,
        )
        assert len(model.gates) == _LAYERS
        gate_ids = [id(g) for g in model.gates]
        assert len(set(gate_ids)) == _LAYERS, "Gates must be distinct objects"

    def test_is_nn_module(self, glu_factory: Callable) -> None:
        """GatedMLP is an nn.Module."""
        model = GatedMLP(
            in_features=_IN,
            out_features=_OUT,
            hidden_size=_HIDDEN,
            num_layers=_LAYERS,
            gate_factory=glu_factory,
        )
        assert isinstance(model, nn.Module)
