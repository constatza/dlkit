from __future__ import annotations

import torch

from dlkit.runtime.pipelines.naming import TargetNameByShapeNamer
from dlkit.runtime.pipelines.pipeline import OutputNamingStep, ProcessingContext


def _randn(*shape: int) -> torch.Tensor:
    return torch.randn(*shape)


def test_target_name_by_shape_single_target_renames() -> None:
    t = _randn(8)
    p = _randn(8)
    namer = TargetNameByShapeNamer()

    preds = {"output": p}
    targets = {"y": t}

    out = namer.rename_predictions(preds, targets)
    assert list(out.keys()) == ["y"]
    assert torch.equal(out["y"], p)


def test_target_name_by_shape_multi_target_renames_by_matching_shape() -> None:
    p1 = _randn(2, 3)
    p2 = _randn(4)
    t1 = _randn(2, 3)
    t2 = _randn(4)
    namer = TargetNameByShapeNamer()

    preds = {"o1": p1, "o2": p2}
    targets = {"y_img": t1, "y_vec": t2}

    out = namer.rename_predictions(preds, targets)
    # Both should map to their matching target names
    assert set(out.keys()) == {"y_img", "y_vec"}
    assert torch.equal(out["y_img"], p1)
    assert torch.equal(out["y_vec"], p2)


def test_target_name_by_shape_keeps_unmatched_prediction_name() -> None:
    p = _randn(5)  # shape does not match
    t = _randn(6)
    namer = TargetNameByShapeNamer()

    out = namer.rename_predictions({"output": p}, {"y": t})
    assert list(out.keys()) == ["output"]
    assert torch.equal(out["output"], p)


def test_target_name_by_shape_duplicate_target_shapes_uses_first_and_keeps_others() -> None:
    # Two targets with identical shapes; strategy keeps first mapping and avoids collisions
    p1 = _randn(3)
    p2 = _randn(3)
    y1 = _randn(3)
    y2 = _randn(3)
    namer = TargetNameByShapeNamer()

    out = namer.rename_predictions({"o1": p1, "o2": p2}, {"y": y1, "z": y2})
    # Deterministic behavior: first target name ('y') is used once; second prediction keeps its name
    assert set(out.keys()) == {"y", "o2"}
    assert torch.equal(out["y"], p1)
    assert torch.equal(out["o2"], p2)


def test_output_naming_step_applies_namer_to_context() -> None:
    # Build a context with predictions and targets and ensure naming step updates keys
    t = _randn(4)
    p = _randn(4)
    ctx = ProcessingContext()
    ctx.targets = {"y": t}
    ctx.predictions = {"output": p}

    step = OutputNamingStep(TargetNameByShapeNamer())
    out = step.process(ctx)

    assert list(out.predictions.keys()) == ["y"]
    assert torch.equal(out.predictions["y"], p)
