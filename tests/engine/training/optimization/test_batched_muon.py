"""Tests for BatchedMuon optimizer."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dlkit.engine.training.optimization.batched_muon import (
    BatchedMuon,
    _batch_zeropower_via_newtonschulz,
)

_NS_COEFF = (3.4445, -4.7750, 2.0315)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mixed_shape_model() -> nn.Sequential:
    """Model with 2-D weights of varying shapes for shape-grouping tests.

    Returns:
        Sequential with Linear(64, 64), Linear(64, 32), Linear(32, 16).
    """
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(64, 64, bias=True),
        nn.Linear(64, 32, bias=True),
        nn.Linear(32, 16, bias=True),
    )


@pytest.fixture()
def muon_params(mixed_shape_model: nn.Sequential) -> list[torch.Tensor]:
    """2-D weight parameters eligible for Muon.

    Args:
        mixed_shape_model: The mixed-shape model fixture.

    Returns:
        List of 2-D weight tensors (excluding biases).
    """
    return [p for p in mixed_shape_model.parameters() if p.ndim == 2]


@pytest.fixture()
def batched_muon(muon_params: list[torch.Tensor]) -> BatchedMuon:
    """BatchedMuon configured over the muon_params fixture.

    Args:
        muon_params: 2-D weight tensors.

    Returns:
        BatchedMuon optimizer instance with lr=1e-3, weight_decay=0.
    """
    return BatchedMuon(muon_params, lr=1e-3, weight_decay=0.0)


# ---------------------------------------------------------------------------
# _batch_zeropower_via_newtonschulz unit tests
# ---------------------------------------------------------------------------


def test_ns_output_shape_preserved() -> None:
    """Output tensors have the same shapes as the inputs."""
    grads = [torch.randn(4, 8), torch.randn(8, 4), torch.randn(16, 16)]
    out = _batch_zeropower_via_newtonschulz(grads, _NS_COEFF, ns_steps=5, eps=1e-7)
    assert len(out) == len(grads)
    for g, o in zip(grads, out, strict=True):
        assert o.shape == g.shape


def test_ns_single_param_matches_reference() -> None:
    """Batched NS on a single param agrees with PyTorch's serial NS."""
    from torch.optim._muon import _zeropower_via_newtonschulz

    torch.manual_seed(0)
    g = torch.randn(32, 32)

    ref = _zeropower_via_newtonschulz(g, _NS_COEFF, ns_steps=5, eps=1e-7)
    (batched,) = _batch_zeropower_via_newtonschulz([g], _NS_COEFF, ns_steps=5, eps=1e-7)

    # Both compute in bfloat16; results should be identical.
    assert torch.allclose(ref.float(), batched.float(), atol=1e-4)


def test_ns_tall_matrix_shape() -> None:
    """Tall matrices (rows > cols) are transposed internally and restored."""
    g = torch.randn(64, 32)  # tall → transposed to (32, 64) inside NS
    (out,) = _batch_zeropower_via_newtonschulz([g], _NS_COEFF, ns_steps=5, eps=1e-7)
    assert out.shape == g.shape


# ---------------------------------------------------------------------------
# BatchedMuon step tests
# ---------------------------------------------------------------------------


def test_step_runs(batched_muon: BatchedMuon, muon_params: list[torch.Tensor]) -> None:
    """step() executes without error when all params have gradients."""
    for p in muon_params:
        p.grad = torch.randn_like(p)
    batched_muon.step()


def test_step_updates_params(batched_muon: BatchedMuon, muon_params: list[torch.Tensor]) -> None:
    """All 2-D weight tensors are modified after a step with non-zero gradients."""
    before = [p.detach().clone() for p in muon_params]
    for p in muon_params:
        p.grad = torch.randn_like(p)
    batched_muon.step()
    for b, p in zip(before, muon_params, strict=True):
        assert not torch.allclose(b, p), f"param {p.shape} was not updated"


def test_step_skips_params_without_grad(
    batched_muon: BatchedMuon, muon_params: list[torch.Tensor]
) -> None:
    """Params without gradients are not modified."""
    before = [p.detach().clone() for p in muon_params]
    # Leave all grads as None
    batched_muon.step()
    for b, p in zip(before, muon_params, strict=True):
        assert torch.allclose(b, p), "param updated despite having no gradient"


def test_step_with_closure(batched_muon: BatchedMuon, muon_params: list[torch.Tensor]) -> None:
    """step(closure) calls the closure and returns the loss."""
    called = [False]

    def closure() -> float:
        called[0] = True
        for p in muon_params:
            p.grad = torch.randn_like(p)
        return 1.23

    loss = batched_muon.step(closure=closure)
    assert called[0], "closure was not invoked"
    assert loss is not None and abs(float(loss) - 1.23) < 1e-5


def test_step_matches_reference_muon(muon_params: list[torch.Tensor]) -> None:
    """BatchedMuon and torch.optim.Muon produce identical updates (bfloat16 tolerance)."""
    params_ref = [p.detach().clone().requires_grad_(True) for p in muon_params]
    params_bat = [p.detach().clone().requires_grad_(True) for p in muon_params]
    grads = [torch.randn_like(p) for p in muon_params]

    for p, g in zip(params_ref, grads, strict=True):
        p.grad = g.clone()
    for p, g in zip(params_bat, grads, strict=True):
        p.grad = g.clone()

    ref = torch.optim.Muon(params_ref, lr=1e-3, weight_decay=0.0)
    bat = BatchedMuon(params_bat, lr=1e-3, weight_decay=0.0)
    ref.step()
    bat.step()

    for a, b in zip(params_ref, params_bat, strict=True):
        max_diff = (a - b).abs().max().item()
        assert torch.allclose(a, b, rtol=1e-2, atol=1e-4), (
            f"BatchedMuon diverges from reference for shape {a.shape}: max diff {max_diff:.6f}"
        )


def test_rejects_non_2d_params() -> None:
    """Constructor raises ValueError for 1-D parameters (same as torch.optim.Muon)."""
    bias = nn.Parameter(torch.randn(8))
    with pytest.raises(ValueError, match="2D"):
        BatchedMuon([bias], lr=1e-3)


def test_momentum_buffer_accumulated(
    muon_params: list[torch.Tensor],
) -> None:
    """Momentum buffer is initialised on first step and persists across steps."""
    opt = BatchedMuon(muon_params, lr=1e-3, weight_decay=0.0, momentum=0.9)
    for p in muon_params:
        p.grad = torch.randn_like(p)
    opt.step()

    # Buffer should now exist for every param.
    for p in muon_params:
        assert "momentum_buffer" in opt.state[p]

    buf_after_1 = [opt.state[p]["momentum_buffer"].clone() for p in muon_params]

    for p in muon_params:
        p.grad = torch.randn_like(p)
    opt.step()

    for p, b1 in zip(muon_params, buf_after_1, strict=True):
        assert not torch.allclose(opt.state[p]["momentum_buffer"], b1), (
            "momentum buffer did not update on second step"
        )
