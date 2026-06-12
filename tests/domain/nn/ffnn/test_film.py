"""Tests for FiLM-conditioned feedforward networks.

Covers FiLMBlock, FiLMResidualBlock, FiLMFFNN, VarWidthFiLMFFNN,
FiLMEmbeddedFFNN, ScaleEquivariantFiLMFFNN, ScaleEquivariantVarWidthFiLMFFNN,
and ScaleEquivariantFiLMEmbeddedFFNN.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from dlkit.domain.nn.contracts import TabulaRSpec
from dlkit.domain.nn.ffnn import (
    FiLMBlock,
    FiLMEmbeddedFFNN,
    FiLMFFNN,
    FiLMResidualBlock,
    ScaleEquivariantFiLMEmbeddedFFNN,
    ScaleEquivariantFiLMFFNN,
    ScaleEquivariantVarWidthFiLMFFNN,
    VarWidthFiLMFFNN,
)

# ── Constants ──────────────────────────────────────────────────────────────────

BATCH = 5
IN_FEATURES = 8
OUT_FEATURES = 4
CONDITION_DIM = 6
HIDDEN_SIZE = 16
NUM_LAYERS = 2
LAYERS_NARROW = [32, 16, 8]
SEED = 99
EQUIVARIANCE_ATOL = 1e-4


# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def feature_input() -> Tensor:
    """Batched feature tensor of shape (BATCH, IN_FEATURES)."""
    torch.manual_seed(SEED)
    return torch.randn(BATCH, IN_FEATURES)


@pytest.fixture
def condition_input() -> Tensor:
    """Batched condition tensor of shape (BATCH, CONDITION_DIM)."""
    torch.manual_seed(SEED + 1)
    return torch.randn(BATCH, CONDITION_DIM)


@pytest.fixture
def square_feature_input() -> Tensor:
    """Square-shaped input (BATCH, HIDDEN_SIZE) for residual blocks."""
    torch.manual_seed(SEED + 2)
    return torch.randn(BATCH, HIDDEN_SIZE)


@pytest.fixture
def square_condition_input() -> Tensor:
    """Condition tensor matching square residual tests."""
    torch.manual_seed(SEED + 3)
    return torch.randn(BATCH, CONDITION_DIM)


@pytest.fixture
def tabular_contract() -> TabulaRSpec:
    """TabulaRSpec with IN_FEATURES -> OUT_FEATURES."""
    return TabulaRSpec(in_shape=(IN_FEATURES,), out_shape=(OUT_FEATURES,))


# ── FiLMBlock fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def film_block() -> FiLMBlock:
    """Standard FiLMBlock(IN_FEATURES -> HIDDEN_SIZE, CONDITION_DIM)."""
    return FiLMBlock(
        in_features=IN_FEATURES,
        out_features=HIDDEN_SIZE,
        condition_dim=CONDITION_DIM,
    )


# ── FiLMResidualBlock fixtures ─────────────────────────────────────────────────


@pytest.fixture
def film_residual_block() -> FiLMResidualBlock:
    """FiLMResidualBlock with feature_dim=HIDDEN_SIZE."""
    return FiLMResidualBlock(feature_dim=HIDDEN_SIZE, condition_dim=CONDITION_DIM)


# ── FiLMFFNN fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def film_ffnn() -> FiLMFFNN:
    """FiLMFFNN with constant width HIDDEN_SIZE and NUM_LAYERS."""
    return FiLMFFNN(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        condition_dim=CONDITION_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )


@pytest.fixture
def film_ffnn_from_contract(tabular_contract: TabulaRSpec) -> FiLMFFNN:
    """FiLMFFNN constructed via from_contract."""
    return FiLMFFNN.from_contract(
        tabular_contract,
        condition_dim=CONDITION_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )


@pytest.fixture
def film_ffnn_single_layer_factory() -> Callable[[], FiLMFFNN]:
    """Callable that constructs a FiLMFFNN with num_layers=1.

    Returns:
        A zero-argument callable whose invocation raises ValueError.
    """
    return lambda: FiLMFFNN(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        condition_dim=CONDITION_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=1,
    )


@pytest.fixture
def film_ffnn_zero_layers_factory() -> Callable[[], FiLMFFNN]:
    """Callable that constructs a FiLMFFNN with num_layers=0.

    Returns:
        A zero-argument callable whose invocation raises ValueError.
    """
    return lambda: FiLMFFNN(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        condition_dim=CONDITION_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=0,
    )


# ── VarWidthFiLMFFNN fixtures ──────────────────────────────────────────────────


@pytest.fixture
def varwidth_film_ffnn() -> VarWidthFiLMFFNN:
    """VarWidthFiLMFFNN with variable-width layers [32, 16, 8]."""
    return VarWidthFiLMFFNN(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        condition_dim=CONDITION_DIM,
        layers=LAYERS_NARROW,
    )


@pytest.fixture
def varwidth_film_ffnn_from_contract(tabular_contract: TabulaRSpec) -> VarWidthFiLMFFNN:
    """VarWidthFiLMFFNN constructed via from_contract."""
    return VarWidthFiLMFFNN.from_contract(
        tabular_contract,
        condition_dim=CONDITION_DIM,
        layers=[HIDDEN_SIZE, HIDDEN_SIZE],
    )


@pytest.fixture
def varwidth_film_ffnn_empty_layers_factory() -> Callable[[], VarWidthFiLMFFNN]:
    """Callable that constructs a VarWidthFiLMFFNN with an empty layers list.

    Returns:
        A zero-argument callable whose invocation raises ValueError.
    """
    return lambda: VarWidthFiLMFFNN(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        condition_dim=CONDITION_DIM,
        layers=[],
    )


@pytest.fixture
def varwidth_film_ffnn_single_layer_factory() -> Callable[[], VarWidthFiLMFFNN]:
    """Callable that constructs a VarWidthFiLMFFNN with a single-element layers list.

    Returns:
        A zero-argument callable whose invocation raises ValueError.
    """
    return lambda: VarWidthFiLMFFNN(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        condition_dim=CONDITION_DIM,
        layers=[HIDDEN_SIZE],
    )


# ── FiLMEmbeddedFFNN fixtures ──────────────────────────────────────────────────


@pytest.fixture
def film_embedded_ffnn() -> FiLMEmbeddedFFNN:
    """FiLMEmbeddedFFNN with hidden_size=HIDDEN_SIZE and num_layers=NUM_LAYERS."""
    return FiLMEmbeddedFFNN(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        condition_dim=CONDITION_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )


@pytest.fixture
def film_embedded_ffnn_from_contract(tabular_contract: TabulaRSpec) -> FiLMEmbeddedFFNN:
    """FiLMEmbeddedFFNN constructed via from_contract."""
    return FiLMEmbeddedFFNN.from_contract(
        tabular_contract,
        condition_dim=CONDITION_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )


@pytest.fixture
def film_embedded_ffnn_square() -> FiLMEmbeddedFFNN:
    """FiLMEmbeddedFFNN with in==out for identity-skip E2E test."""
    return FiLMEmbeddedFFNN(
        in_features=IN_FEATURES,
        out_features=IN_FEATURES,
        condition_dim=CONDITION_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )


@pytest.fixture
def film_embedded_ffnn_zero_layers_factory() -> Callable[[], FiLMEmbeddedFFNN]:
    """Callable that constructs a FiLMEmbeddedFFNN with num_layers=0.

    Returns:
        A zero-argument callable whose invocation raises ValueError.
    """
    return lambda: FiLMEmbeddedFFNN(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        condition_dim=CONDITION_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=0,
    )


# ── ScaleEquivariantFiLMFFNN fixtures ──────────────────────────────────────────


@pytest.fixture
def se_film_ffnn() -> ScaleEquivariantFiLMFFNN:
    """ScaleEquivariantFiLMFFNN with constant width HIDDEN_SIZE and NUM_LAYERS."""
    return ScaleEquivariantFiLMFFNN(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        condition_dim=CONDITION_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )


# ── ScaleEquivariantVarWidthFiLMFFNN fixtures ─────────────────────────────────


@pytest.fixture
def se_varwidth_film_ffnn() -> ScaleEquivariantVarWidthFiLMFFNN:
    """ScaleEquivariantVarWidthFiLMFFNN with variable-width layers [32, 16, 8]."""
    return ScaleEquivariantVarWidthFiLMFFNN(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        condition_dim=CONDITION_DIM,
        layers=LAYERS_NARROW,
    )


# ── ScaleEquivariantFiLMEmbeddedFFNN fixtures ──────────────────────────────────


@pytest.fixture
def se_film_embedded_ffnn() -> ScaleEquivariantFiLMEmbeddedFFNN:
    """ScaleEquivariantFiLMEmbeddedFFNN."""
    return ScaleEquivariantFiLMEmbeddedFFNN(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        condition_dim=CONDITION_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )


@pytest.fixture
def se_film_ffnn_from_contract(tabular_contract: TabulaRSpec) -> ScaleEquivariantFiLMFFNN:
    """ScaleEquivariantFiLMFFNN constructed via from_contract."""
    return ScaleEquivariantFiLMFFNN.from_contract(
        tabular_contract,
        condition_dim=CONDITION_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )


@pytest.fixture
def se_varwidth_film_ffnn_from_contract(
    tabular_contract: TabulaRSpec,
) -> ScaleEquivariantVarWidthFiLMFFNN:
    """ScaleEquivariantVarWidthFiLMFFNN constructed via from_contract."""
    return ScaleEquivariantVarWidthFiLMFFNN.from_contract(
        tabular_contract,
        condition_dim=CONDITION_DIM,
        layers=[HIDDEN_SIZE, HIDDEN_SIZE],
    )


@pytest.fixture
def se_film_embedded_ffnn_from_contract(
    tabular_contract: TabulaRSpec,
) -> ScaleEquivariantFiLMEmbeddedFFNN:
    """ScaleEquivariantFiLMEmbeddedFFNN constructed via from_contract."""
    return ScaleEquivariantFiLMEmbeddedFFNN.from_contract(
        tabular_contract,
        condition_dim=CONDITION_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )


# ── FiLMBlock tests ────────────────────────────────────────────────────────────


def test_film_block_output_shape(
    film_block: FiLMBlock,
    feature_input: Tensor,
    condition_input: Tensor,
) -> None:
    """FiLMBlock output shape is (BATCH, HIDDEN_SIZE)."""
    with torch.no_grad():
        out = film_block(feature_input, condition_input)
    assert out.shape == (BATCH, HIDDEN_SIZE)


def test_film_block_forward_runs_without_error(
    film_block: FiLMBlock,
    feature_input: Tensor,
    condition_input: Tensor,
) -> None:
    """FiLMBlock.forward executes end-to-end without raising."""
    with torch.no_grad():
        out = film_block(feature_input, condition_input)
    assert isinstance(out, Tensor)


# ── FiLMResidualBlock tests ────────────────────────────────────────────────────


def test_film_residual_block_output_shape(
    film_residual_block: FiLMResidualBlock,
    square_feature_input: Tensor,
    square_condition_input: Tensor,
) -> None:
    """FiLMResidualBlock output shape equals input shape (square residual)."""
    with torch.no_grad():
        out = film_residual_block(square_feature_input, square_condition_input)
    assert out.shape == square_feature_input.shape


def test_film_residual_block_forward_runs_without_error(
    film_residual_block: FiLMResidualBlock,
    square_feature_input: Tensor,
    square_condition_input: Tensor,
) -> None:
    """FiLMResidualBlock.forward executes end-to-end without raising."""
    with torch.no_grad():
        out = film_residual_block(square_feature_input, square_condition_input)
    assert isinstance(out, Tensor)


# ── FiLMFFNN tests ─────────────────────────────────────────────────────────────


def test_film_ffnn_output_shape(
    film_ffnn: FiLMFFNN,
    feature_input: Tensor,
    condition_input: Tensor,
) -> None:
    """FiLMFFNN output shape is (BATCH, OUT_FEATURES)."""
    with torch.no_grad():
        out = film_ffnn(feature_input, condition_input)
    assert out.shape == (BATCH, OUT_FEATURES)


def test_film_ffnn_from_contract_output_shape(
    film_ffnn_from_contract: FiLMFFNN,
    feature_input: Tensor,
    condition_input: Tensor,
) -> None:
    """from_contract FiLMFFNN produces correct output shape."""
    with torch.no_grad():
        out = film_ffnn_from_contract(feature_input, condition_input)
    assert out.shape == (BATCH, OUT_FEATURES)


def test_film_ffnn_from_contract_uses_contract_dimensions(
    film_ffnn_from_contract: FiLMFFNN,
    tabular_contract: TabulaRSpec,
) -> None:
    """from_contract creates a network with in_features and out_features from the contract."""
    assert film_ffnn_from_contract.embed.in_features == tabular_contract.in_shape[0]
    assert film_ffnn_from_contract.head.out_features == tabular_contract.out_shape[0]


def test_film_ffnn_zero_layers_raises(
    film_ffnn_zero_layers_factory: Callable[[], FiLMFFNN],
) -> None:
    """FiLMFFNN with num_layers=0 raises ValueError."""
    with pytest.raises(ValueError):
        film_ffnn_zero_layers_factory()


def test_film_ffnn_single_layer_raises(
    film_ffnn_single_layer_factory: Callable[[], FiLMFFNN],
) -> None:
    """FiLMFFNN with num_layers=1 raises ValueError."""
    with pytest.raises(ValueError):
        film_ffnn_single_layer_factory()


# ── VarWidthFiLMFFNN tests ─────────────────────────────────────────────────────


def test_varwidth_film_ffnn_output_shape(
    varwidth_film_ffnn: VarWidthFiLMFFNN,
    feature_input: Tensor,
    condition_input: Tensor,
) -> None:
    """VarWidthFiLMFFNN output shape is (BATCH, OUT_FEATURES)."""
    with torch.no_grad():
        out = varwidth_film_ffnn(feature_input, condition_input)
    assert out.shape == (BATCH, OUT_FEATURES)


def test_varwidth_film_ffnn_from_contract_output_shape(
    varwidth_film_ffnn_from_contract: VarWidthFiLMFFNN,
    feature_input: Tensor,
    condition_input: Tensor,
) -> None:
    """from_contract VarWidthFiLMFFNN produces correct output shape."""
    with torch.no_grad():
        out = varwidth_film_ffnn_from_contract(feature_input, condition_input)
    assert out.shape == (BATCH, OUT_FEATURES)


def test_varwidth_film_ffnn_empty_layers_raises(
    varwidth_film_ffnn_empty_layers_factory: Callable[[], VarWidthFiLMFFNN],
) -> None:
    """VarWidthFiLMFFNN with empty layers list raises ValueError."""
    with pytest.raises(ValueError):
        varwidth_film_ffnn_empty_layers_factory()


def test_varwidth_film_ffnn_single_layer_raises(
    varwidth_film_ffnn_single_layer_factory: Callable[[], VarWidthFiLMFFNN],
) -> None:
    """VarWidthFiLMFFNN with a single-element layers list raises ValueError."""
    with pytest.raises(ValueError):
        varwidth_film_ffnn_single_layer_factory()


# ── FiLMEmbeddedFFNN tests ─────────────────────────────────────────────────────


def test_film_embedded_ffnn_output_shape(
    film_embedded_ffnn: FiLMEmbeddedFFNN,
    feature_input: Tensor,
    condition_input: Tensor,
) -> None:
    """FiLMEmbeddedFFNN output shape is (BATCH, OUT_FEATURES)."""
    with torch.no_grad():
        out = film_embedded_ffnn(feature_input, condition_input)
    assert out.shape == (BATCH, OUT_FEATURES)


def test_film_embedded_ffnn_from_contract_output_shape(
    film_embedded_ffnn_from_contract: FiLMEmbeddedFFNN,
    feature_input: Tensor,
    condition_input: Tensor,
) -> None:
    """from_contract FiLMEmbeddedFFNN produces correct output shape."""
    with torch.no_grad():
        out = film_embedded_ffnn_from_contract(feature_input, condition_input)
    assert out.shape == (BATCH, OUT_FEATURES)


def test_film_embedded_ffnn_from_contract_uses_contract_dimensions(
    film_embedded_ffnn_from_contract: FiLMEmbeddedFFNN,
    tabular_contract: TabulaRSpec,
) -> None:
    """from_contract creates a network with dimensions matching the contract."""
    assert film_embedded_ffnn_from_contract.embed.in_features == tabular_contract.in_shape[0]
    assert film_embedded_ffnn_from_contract.head.out_features == tabular_contract.out_shape[0]


def test_film_embedded_ffnn_e2e_skip_works(
    film_embedded_ffnn_square: FiLMEmbeddedFFNN,
    feature_input: Tensor,
    condition_input: Tensor,
) -> None:
    """FiLMEmbeddedFFNN with in==out runs without error (E2E skip path active)."""
    with torch.no_grad():
        out = film_embedded_ffnn_square(feature_input, condition_input)
    assert out.shape == (BATCH, IN_FEATURES)


def test_film_embedded_ffnn_zero_layers_raises(
    film_embedded_ffnn_zero_layers_factory: Callable[[], FiLMEmbeddedFFNN],
) -> None:
    """FiLMEmbeddedFFNN with num_layers=0 raises ValueError."""
    with pytest.raises(ValueError):
        film_embedded_ffnn_zero_layers_factory()


# ── ScaleEquivariantFiLMFFNN tests ─────────────────────────────────────────────


def test_se_film_ffnn_output_shape(
    se_film_ffnn: ScaleEquivariantFiLMFFNN,
    feature_input: Tensor,
    condition_input: Tensor,
) -> None:
    """ScaleEquivariantFiLMFFNN output shape is (BATCH, OUT_FEATURES)."""
    with torch.no_grad():
        out = se_film_ffnn(feature_input, condition_input)
    assert out.shape == (BATCH, OUT_FEATURES)


@pytest.mark.parametrize("alpha", [0.5, 2.0, 10.0])
def test_se_film_ffnn_scale_equivariance(
    se_film_ffnn: ScaleEquivariantFiLMFFNN,
    feature_input: Tensor,
    condition_input: Tensor,
    alpha: float,
) -> None:
    """f(alpha*x, c) == alpha * f(x, c) for positive scalars alpha."""
    se_film_ffnn.eval()
    with torch.no_grad():
        out_x = se_film_ffnn(feature_input, condition_input)
        out_ax = se_film_ffnn(alpha * feature_input, condition_input)
    assert torch.allclose(out_ax, alpha * out_x, atol=EQUIVARIANCE_ATOL), (
        f"Scale equivariance failed for alpha={alpha}: "
        f"max diff = {(out_ax - alpha * out_x).abs().max().item():.6f}"
    )


def test_se_film_ffnn_from_contract_output_shape(
    se_film_ffnn_from_contract: ScaleEquivariantFiLMFFNN,
    feature_input: Tensor,
    condition_input: Tensor,
) -> None:
    """ScaleEquivariantFiLMFFNN.from_contract produces correct output shape."""
    with torch.no_grad():
        out = se_film_ffnn_from_contract(feature_input, condition_input)
    assert out.shape == (BATCH, OUT_FEATURES)


# ── ScaleEquivariantVarWidthFiLMFFNN tests ────────────────────────────────────


def test_se_varwidth_film_ffnn_output_shape(
    se_varwidth_film_ffnn: ScaleEquivariantVarWidthFiLMFFNN,
    feature_input: Tensor,
    condition_input: Tensor,
) -> None:
    """ScaleEquivariantVarWidthFiLMFFNN output shape is (BATCH, OUT_FEATURES)."""
    with torch.no_grad():
        out = se_varwidth_film_ffnn(feature_input, condition_input)
    assert out.shape == (BATCH, OUT_FEATURES)


@pytest.mark.parametrize("alpha", [0.5, 2.0, 10.0])
def test_se_varwidth_film_ffnn_scale_equivariance(
    se_varwidth_film_ffnn: ScaleEquivariantVarWidthFiLMFFNN,
    feature_input: Tensor,
    condition_input: Tensor,
    alpha: float,
) -> None:
    """f(alpha*x, c) == alpha * f(x, c) for positive scalars alpha."""
    se_varwidth_film_ffnn.eval()
    with torch.no_grad():
        out_x = se_varwidth_film_ffnn(feature_input, condition_input)
        out_ax = se_varwidth_film_ffnn(alpha * feature_input, condition_input)
    assert torch.allclose(out_ax, alpha * out_x, atol=EQUIVARIANCE_ATOL), (
        f"Scale equivariance failed for alpha={alpha}: "
        f"max diff = {(out_ax - alpha * out_x).abs().max().item():.6f}"
    )


def test_se_varwidth_film_ffnn_from_contract_output_shape(
    se_varwidth_film_ffnn_from_contract: ScaleEquivariantVarWidthFiLMFFNN,
    feature_input: Tensor,
    condition_input: Tensor,
) -> None:
    """ScaleEquivariantVarWidthFiLMFFNN.from_contract produces correct output shape."""
    with torch.no_grad():
        out = se_varwidth_film_ffnn_from_contract(feature_input, condition_input)
    assert out.shape == (BATCH, OUT_FEATURES)


# ── ScaleEquivariantFiLMEmbeddedFFNN tests ─────────────────────────────────────


def test_se_film_embedded_ffnn_output_shape(
    se_film_embedded_ffnn: ScaleEquivariantFiLMEmbeddedFFNN,
    feature_input: Tensor,
    condition_input: Tensor,
) -> None:
    """ScaleEquivariantFiLMEmbeddedFFNN output shape is (BATCH, OUT_FEATURES)."""
    with torch.no_grad():
        out = se_film_embedded_ffnn(feature_input, condition_input)
    assert out.shape == (BATCH, OUT_FEATURES)


@pytest.mark.parametrize("alpha", [0.5, 2.0, 10.0])
def test_se_film_embedded_ffnn_scale_equivariance(
    se_film_embedded_ffnn: ScaleEquivariantFiLMEmbeddedFFNN,
    feature_input: Tensor,
    condition_input: Tensor,
    alpha: float,
) -> None:
    """f(alpha*x, c) == alpha * f(x, c) for positive scalars alpha."""
    se_film_embedded_ffnn.eval()
    with torch.no_grad():
        out_x = se_film_embedded_ffnn(feature_input, condition_input)
        out_ax = se_film_embedded_ffnn(alpha * feature_input, condition_input)
    assert torch.allclose(out_ax, alpha * out_x, atol=EQUIVARIANCE_ATOL), (
        f"Scale equivariance failed for alpha={alpha}: "
        f"max diff = {(out_ax - alpha * out_x).abs().max().item():.6f}"
    )


def test_se_film_embedded_ffnn_from_contract_output_shape(
    se_film_embedded_ffnn_from_contract: ScaleEquivariantFiLMEmbeddedFFNN,
    feature_input: Tensor,
    condition_input: Tensor,
) -> None:
    """ScaleEquivariantFiLMEmbeddedFFNN.from_contract produces correct output shape."""
    with torch.no_grad():
        out = se_film_embedded_ffnn_from_contract(feature_input, condition_input)
    assert out.shape == (BATCH, OUT_FEATURES)
