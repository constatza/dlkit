from __future__ import annotations

import inspect
import math
from typing import Any, cast

import pytest
import torch
from torch import nn

from dlkit.common.shapes import ShapeContext
from dlkit.domain.nn.ffnn.constrained import EmbeddedFactorizedFFNN, FactorizedFFNN
from dlkit.domain.nn.primitives import FactorizedLinear, SkipConnection

from .conftest import VarianceBand

ShapeMapping = dict[str, tuple[int, ...]]

#: EmbeddedFactorizedFFNN(project=False) body, measured ~1.45-1.78 at depths 8-64.
STD_RATIO_BAND = VarianceBand(1.2, 2.2)


def _unwrap_factorized_layer(module: nn.Module) -> FactorizedLinear:
    if isinstance(module, SkipConnection):
        module = cast(Any, module).module
    layer = cast(Any, module).layer
    assert isinstance(layer, FactorizedLinear)
    return layer


@pytest.fixture
def factorized_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    return {"x": (3,)}, {"y": (2,)}


@pytest.mark.parametrize("skip", [True, False])
def test_embedded_factorized_output_shape(skip: bool) -> None:
    model = EmbeddedFactorizedFFNN(
        in_features=3,
        out_features=2,
        hidden_size=4,
        num_layers=2,
        skip=skip,
    )

    assert model(torch.randn(5, 3)).shape == (5, 2)


@pytest.mark.parametrize("skip", [True, False])
def test_embedded_factorized_project_false_reproduces_constant_width_shape(skip: bool) -> None:
    model = EmbeddedFactorizedFFNN(
        in_features=4,
        out_features=4,
        hidden_size=4,
        num_layers=3,
        project=False,
        skip=skip,
    )

    assert isinstance(model.embedding_layer, nn.Identity)
    assert isinstance(model.regression_layer, nn.Identity)
    assert model(torch.randn(5, 4)).shape == (5, 4)


def test_embedded_factorized_project_false_rejects_rectangular_shape() -> None:
    with pytest.raises(ValueError, match="project=False"):
        EmbeddedFactorizedFFNN(
            in_features=3,
            out_features=2,
            hidden_size=4,
            num_layers=2,
            project=False,
        )


@pytest.mark.parametrize("skip", [True, False])
def test_embedded_factorized_body_has_skip_iff_requested(skip: bool) -> None:
    model = EmbeddedFactorizedFFNN(
        in_features=3,
        out_features=2,
        hidden_size=4,
        num_layers=2,
        skip=skip,
    )

    assert isinstance(cast(Any, model).body.blocks[0], SkipConnection) is skip


@pytest.mark.parametrize("skip", [True, False])
def test_embedded_factorized_from_context(
    skip: bool,
    factorized_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = factorized_shapes
    model = EmbeddedFactorizedFFNN.from_context(
        ShapeContext(in_shapes, out_shapes),
        hidden_size=4,
        num_layers=2,
        skip=skip,
    )

    assert model(torch.randn(4, in_shapes["x"][0])).shape == (4, out_shapes["y"][0])


def test_embedded_factorized_all_owned_linear_kernels_are_factorized() -> None:
    model = EmbeddedFactorizedFFNN(in_features=3, out_features=2, hidden_size=4, num_layers=2)

    assert isinstance(model.embedding_layer, FactorizedLinear)
    assert isinstance(model.regression_layer, FactorizedLinear)
    _unwrap_factorized_layer(cast(Any, model).body.blocks[0])


@pytest.mark.parametrize("skip", [True, False])
def test_factorized_ffnn_output_shape(skip: bool) -> None:
    model = FactorizedFFNN(
        in_features=4,
        out_features=2,
        hidden_size=8,
        num_layers=3,
        skip=skip,
    )

    assert model(torch.randn(5, 4)).shape == (5, 2)


@pytest.mark.parametrize("skip", [True, False])
def test_factorized_ffnn_body_has_skip_iff_requested(skip: bool) -> None:
    model = FactorizedFFNN(
        in_features=4,
        out_features=2,
        hidden_size=8,
        num_layers=3,
        skip=skip,
    )

    assert isinstance(cast(Any, model).body.blocks[0], SkipConnection) is skip


def test_factorized_ffnn_single_layer_has_no_body_blocks() -> None:
    model = FactorizedFFNN(in_features=4, out_features=2, hidden_size=8, num_layers=1)

    assert model(torch.randn(3, 4)).shape == (3, 2)
    assert len(cast(Any, model).body.blocks) == 0


def test_factorized_ffnn_square_case_defaults_hidden_size() -> None:
    model = FactorizedFFNN(in_features=4, out_features=4, num_layers=2)

    assert model(torch.randn(3, 4)).shape == (3, 4)


def test_factorized_linear_default_mean_is_zero() -> None:
    assert inspect.signature(FactorizedLinear.__init__).parameters["mean"].default == 0.0


@pytest.mark.parametrize("num_layers", [8, 16, 32, 64])
def test_deep_embedded_factorized_project_false_output_std_stays_bounded(
    num_layers: int,
) -> None:
    torch.manual_seed(0)
    model = EmbeddedFactorizedFFNN(
        in_features=8,
        out_features=8,
        hidden_size=8,
        num_layers=num_layers,
        project=False,
        skip=True,
        normalize=None,
    )
    model.eval()

    with torch.no_grad():
        x = torch.randn(256, 8)
        y = model(x)

    body = cast(Any, model).body
    assert body.blocks[0].branch_scale == pytest.approx(1 / math.sqrt(num_layers))
    ratio = y.std().item() / x.std().item()
    assert STD_RATIO_BAND.low < ratio < STD_RATIO_BAND.high, (
        f"Output std diverged at {num_layers} layers: {ratio:.2f}x"
    )
