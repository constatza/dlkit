from __future__ import annotations

import pytest
import torch

from dlkit.runtime.pipelines.pipeline import LossPairingStep, ProcessingContext


def _randn(*shape: int) -> torch.Tensor:
    return torch.randn(*shape)


def _ctx(
    features: dict[str, torch.Tensor] | None = None,
    targets: dict[str, torch.Tensor] | None = None,
    predictions: dict[str, torch.Tensor] | None = None,
) -> ProcessingContext:
    c = ProcessingContext()
    c.features = features or {}
    c.targets = targets or {}
    c.predictions = predictions or {}
    return c


def test_strict_pairing_success() -> None:
    t = _randn(8)
    p = _randn(8)
    step = LossPairingStep(entry_configs={})

    ctx = _ctx(targets={"y": t}, predictions={"y": p})
    out = step.process(ctx)

    assert "y" in out.loss_data
    pair = out.loss_data["y"]
    assert isinstance(pair, tuple) and len(pair) == 2
    assert torch.equal(pair[0], p) and torch.equal(pair[1], t)


def test_single_target_single_prediction_fallback() -> None:
    t = _randn(4)
    p = _randn(4)
    step = LossPairingStep(entry_configs={})

    ctx = _ctx(targets={"y": t}, predictions={"output": p})
    out = step.process(ctx)

    pair = out.loss_data["y"]
    assert isinstance(pair, tuple) and len(pair) == 2
    assert torch.equal(pair[0], p) and torch.equal(pair[1], t)


def test_missing_prediction_error_message() -> None:
    t_y = _randn(2)
    t_z = _randn(2)
    step = LossPairingStep(entry_configs={})

    ctx = _ctx(targets={"y": t_y, "z": t_z}, predictions={"y": _randn(2)})
    with pytest.raises(RuntimeError) as err:
        step.process(ctx)
    msg = str(err.value)
    assert "missing predictions for targets" in msg
    assert "['z']" in msg
    assert "available targets=['y', 'z']" in msg
    assert "predictions=['y']" in msg


def test_unexpected_prediction_error_message() -> None:
    t = _randn(3)
    step = LossPairingStep(entry_configs={})

    ctx = _ctx(targets={"y": t}, predictions={"y": _randn(3), "junk": _randn(3)})
    with pytest.raises(RuntimeError) as err:
        step.process(ctx)
    msg = str(err.value)
    assert "unexpected prediction keys" in msg
    assert "['junk']" in msg
    assert "available targets=['y']" in msg
    assert "predictions=['y', 'junk']" in msg


def test_autoencoder_uses_features_as_targets() -> None:
    x = _randn(5, 4)
    p = _randn(5, 4)
    # Autoencoder: treat features as targets when targets are absent
    step = LossPairingStep(entry_configs={}, is_autoencoder=True)

    ctx = _ctx(features={"x": x}, targets={}, predictions={"x": p})
    out = step.process(ctx)

    pair = out.loss_data["x"]
    assert isinstance(pair, tuple) and len(pair) == 2
    assert torch.equal(pair[0], p) and torch.equal(pair[1], x)
