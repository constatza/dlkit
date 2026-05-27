from __future__ import annotations

import inspect
from typing import Any, cast

import pytest
import torch
from torch import nn

from dlkit.domain.nn import (
    SPDFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedSimpleFactorizedFFNN,
    EmbeddedSimpleSPDFactorizedFFNN,
    EmbeddedSimpleSPDFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedSPDFFNN,
    FactorizedFFNN,
    SimpleFactorizedFFNN,
    SimpleSPDFactorizedFFNN,
    SimpleSPDFFNN,
    SPDFactorizedFFNN,
)
from dlkit.domain.nn.contracts import TabulaRSpec
from dlkit.domain.nn.ffnn.constrained import (
    EmbeddedParametricFFNN,
    EmbeddedSimpleParametricFFNN,
)
from dlkit.domain.nn.primitives import SkipConnection


def _dummy_factory(n: int) -> nn.Module:
    return nn.Linear(n, n)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def spd_contract_square() -> TabulaRSpec:
    return TabulaRSpec(in_shape=(4,), out_shape=(4,))


@pytest.fixture
def spd_contract_nonsquare() -> TabulaRSpec:
    return TabulaRSpec(in_shape=(4,), out_shape=(2,))


@pytest.fixture
def factorized_contract() -> TabulaRSpec:
    return TabulaRSpec(in_shape=(3,), out_shape=(2,))


# ── Embedded SPD (all-SPD, square) ───────────────────────────────────────────

SPD_EMBEDDED_PAIRS = [
    (EmbeddedSPDFFNN, EmbeddedSimpleSPDFFNN),
    (EmbeddedSPDFactorizedFFNN, EmbeddedSimpleSPDFactorizedFFNN),
]

SPD_NONEMBEDDED_PAIRS = [
    (SPDFFNN, SimpleSPDFFNN),
    (SPDFactorizedFFNN, SimpleSPDFactorizedFFNN),
]

FACTORIZED_EMBEDDED_PAIRS = [
    (EmbeddedFactorizedFFNN, EmbeddedSimpleFactorizedFFNN),
]

FACTORIZED_NONEMBEDDED_PAIRS = [
    (FactorizedFFNN, SimpleFactorizedFFNN),
]


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SPD_EMBEDDED_PAIRS)
def test_embedded_spd_variants_produce_correct_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, num_layers=3)
    plain = plain_cls(in_features=4, num_layers=3)
    x = torch.randn(5, 4)
    assert residual(x).shape == (5, 4)
    assert plain(x).shape == (5, 4)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SPD_EMBEDDED_PAIRS)
def test_embedded_spd_variants_body_has_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, num_layers=3)  # 1 body block
    plain = plain_cls(in_features=4, num_layers=3)
    assert isinstance(cast(Any, residual).body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).body.blocks[0], SkipConnection)


@pytest.mark.parametrize(
    "model_cls",
    [
        EmbeddedSPDFFNN,
        EmbeddedSimpleSPDFFNN,
        EmbeddedSPDFactorizedFFNN,
        EmbeddedSimpleSPDFactorizedFFNN,
    ],
)
def test_embedded_spd_variants_require_num_layers_ge_2(
    model_cls: type[nn.Module],
) -> None:
    with pytest.raises(ValueError, match="num_layers >= 2"):
        model_cls(in_features=4, num_layers=1)


@pytest.mark.parametrize(
    "model_cls",
    [
        EmbeddedSPDFFNN,
        EmbeddedSimpleSPDFFNN,
        EmbeddedSPDFactorizedFFNN,
        EmbeddedSimpleSPDFactorizedFFNN,
    ],
)
def test_embedded_spd_from_contract_requires_square(
    model_cls: type[nn.Module],
    spd_contract_nonsquare: TabulaRSpec,
) -> None:
    with pytest.raises(ValueError, match="square contract"):
        cast(Any, model_cls).from_contract(spd_contract_nonsquare, num_layers=3)


@pytest.mark.parametrize(
    "model_cls",
    [
        EmbeddedSPDFFNN,
        EmbeddedSimpleSPDFFNN,
        EmbeddedSPDFactorizedFFNN,
        EmbeddedSimpleSPDFactorizedFFNN,
    ],
)
def test_embedded_spd_from_contract_works_for_square(
    model_cls: type[nn.Module],
    spd_contract_square: TabulaRSpec,
) -> None:
    model = cast(Any, model_cls).from_contract(spd_contract_square, num_layers=3)
    x = torch.randn(4, spd_contract_square.in_shape[0])
    assert model(x).shape == (4, spd_contract_square.out_shape[0])


# ── Non-embedded SPD ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SPD_NONEMBEDDED_PAIRS)
def test_nonembedded_spd_variants_produce_correct_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, num_layers=3)
    plain = plain_cls(in_features=4, num_layers=3)
    x = torch.randn(5, 4)
    assert residual(x).shape == (5, 4)
    assert plain(x).shape == (5, 4)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SPD_NONEMBEDDED_PAIRS)
def test_nonembedded_spd_variants_body_has_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, num_layers=3)  # 2 body blocks
    plain = plain_cls(in_features=4, num_layers=3)
    assert isinstance(cast(Any, residual).body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).body.blocks[0], SkipConnection)


@pytest.mark.parametrize(
    ("cls",), [(SPDFFNN,), (SimpleSPDFFNN,), (SPDFactorizedFFNN,), (SimpleSPDFactorizedFFNN,)]
)
def test_nonembedded_spd_single_layer_has_no_body_blocks(cls: type[nn.Module]) -> None:
    model = cls(in_features=4, num_layers=1)
    x = torch.randn(3, 4)
    assert model(x).shape == (3, 4)
    assert len(cast(Any, model).body.blocks) == 0


@pytest.mark.parametrize(
    "model_cls", [SPDFFNN, SimpleSPDFFNN, SPDFactorizedFFNN, SimpleSPDFactorizedFFNN]
)
def test_nonembedded_spd_from_contract_requires_square(
    model_cls: type[nn.Module],
    spd_contract_nonsquare: TabulaRSpec,
) -> None:
    with pytest.raises(ValueError, match="square contract"):
        cast(Any, model_cls).from_contract(spd_contract_nonsquare, num_layers=2)


# ── Embedded Factorized ───────────────────────────────────────────────────────


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FACTORIZED_EMBEDDED_PAIRS)
def test_embedded_factorized_variants_produce_correct_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    plain = plain_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    x = torch.randn(5, 3)
    assert residual(x).shape == (5, 2)
    assert plain(x).shape == (5, 2)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FACTORIZED_EMBEDDED_PAIRS)
def test_embedded_factorized_variants_body_has_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    plain = plain_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    assert isinstance(cast(Any, residual).body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).body.blocks[0], SkipConnection)


@pytest.mark.parametrize("model_cls", [EmbeddedFactorizedFFNN, EmbeddedSimpleFactorizedFFNN])
def test_embedded_factorized_from_contract(
    model_cls: type[nn.Module],
    factorized_contract: TabulaRSpec,
) -> None:
    model = cast(Any, model_cls).from_contract(factorized_contract, hidden_size=4, num_layers=2)
    x = torch.randn(4, factorized_contract.in_shape[0])
    assert model(x).shape == (4, factorized_contract.out_shape[0])


# ── Non-embedded Factorized ───────────────────────────────────────────────────


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FACTORIZED_NONEMBEDDED_PAIRS)
def test_nonembedded_factorized_variants_produce_correct_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, out_features=2, hidden_size=8, num_layers=3)
    plain = plain_cls(in_features=4, out_features=2, hidden_size=8, num_layers=3)
    x = torch.randn(5, 4)
    assert residual(x).shape == (5, 2)
    assert plain(x).shape == (5, 2)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FACTORIZED_NONEMBEDDED_PAIRS)
def test_nonembedded_factorized_body_has_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, out_features=2, hidden_size=8, num_layers=3)
    plain = plain_cls(in_features=4, out_features=2, hidden_size=8, num_layers=3)
    assert isinstance(cast(Any, residual).body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).body.blocks[0], SkipConnection)


@pytest.mark.parametrize("model_cls", [FactorizedFFNN, SimpleFactorizedFFNN])
def test_nonembedded_factorized_single_layer_has_no_body_blocks(
    model_cls: type[nn.Module],
) -> None:
    model = model_cls(in_features=4, out_features=2, hidden_size=8, num_layers=1)
    x = torch.randn(3, 4)
    assert model(x).shape == (3, 2)
    assert len(cast(Any, model).body.blocks) == 0


@pytest.mark.parametrize("model_cls", [FactorizedFFNN, SimpleFactorizedFFNN])
def test_nonembedded_factorized_from_contract(
    model_cls: type[nn.Module],
    factorized_contract: TabulaRSpec,
) -> None:
    model = cast(Any, model_cls).from_contract(factorized_contract, hidden_size=8, num_layers=2)
    x = torch.randn(4, factorized_contract.in_shape[0])
    assert model(x).shape == (4, factorized_contract.out_shape[0])


@pytest.mark.parametrize("model_cls", [FactorizedFFNN, SimpleFactorizedFFNN])
def test_nonembedded_factorized_square_case_defaults_hidden_size(
    model_cls: type[nn.Module],
) -> None:
    model = model_cls(in_features=4, out_features=4, num_layers=2)
    x = torch.randn(3, 4)
    assert model(x).shape == (3, 4)


# ── Generic parametric builders ───────────────────────────────────────────────


def test_embedded_parametric_ffnn_is_residual() -> None:
    m = EmbeddedParametricFFNN(
        in_features=4,
        out_features=2,
        hidden_size=8,
        num_layers=2,
        layer_factory=_dummy_factory,
    )
    assert isinstance(cast(Any, m).body.blocks[0], SkipConnection)


def test_embedded_simple_parametric_ffnn_is_plain() -> None:
    m = EmbeddedSimpleParametricFFNN(
        in_features=4,
        out_features=2,
        hidden_size=8,
        num_layers=2,
        layer_factory=_dummy_factory,
    )
    assert not isinstance(cast(Any, m).body.blocks[0], SkipConnection)


def test_embedded_parametric_ffnn_has_no_residual_param() -> None:
    sig = inspect.signature(EmbeddedParametricFFNN.__init__)
    assert "residual" not in sig.parameters


def test_embedded_factorized_square_case_uses_in_features_as_hidden_size() -> None:
    m = EmbeddedFactorizedFFNN(in_features=4, out_features=4, num_layers=2)
    assert cast(Any, m).embedding_layer.in_features == 4
    assert cast(Any, m).embedding_layer.out_features == 4
    assert cast(Any, m).regression_layer.out_features == 4
