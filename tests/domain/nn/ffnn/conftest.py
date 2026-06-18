from __future__ import annotations

import pytest
import torch

from dlkit.domain.nn.primitives.gated import GLUGate, GRNGate, SwiGLUGate, UVGate

ShapeMapping = dict[str, tuple[int, ...]]

_IN = 4
_OUT = 2


@pytest.fixture
def glu_factory():
    """Factory producing GLUGate(hidden_size=8)."""
    return lambda: GLUGate(hidden_size=8)


@pytest.fixture
def swiglu_factory():
    """Factory producing SwiGLUGate(hidden_size=8)."""
    return lambda: SwiGLUGate(hidden_size=8)


@pytest.fixture
def grn_factory():
    """Factory producing GRNGate(hidden_size=8) — expects hidden-dim context."""
    return lambda: GRNGate(hidden_size=8)


@pytest.fixture
def grn_compatible_factory():
    """Factory producing GRNGate(hidden_size=8, context_size=4).

    context_size=4 matches the GatedMLP in_features=4 used in FFNN tests.
    """
    return lambda: GRNGate(hidden_size=8, context_size=4)


@pytest.fixture
def uv_factory():
    """Factory producing UVGate(in_features=4, hidden_size=8)."""
    return lambda: UVGate(in_features=4, hidden_size=8)


@pytest.fixture
def tabular_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    """Feature/target shape mappings matching the standard GatedMLP test dimensions."""
    return {"x": (_IN,)}, {"y": (_OUT,)}


@pytest.fixture
def network_input(batch_size: int) -> torch.Tensor:
    """4-feature input for GatedMLP tests.

    Shape: (batch_size, 4)
    """
    return torch.randn(batch_size, 4)
