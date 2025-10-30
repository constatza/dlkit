"""Tests for PrecisionValidationStep correctness."""

from types import SimpleNamespace

import pytest
import torch

from dlkit.runtime.pipelines.context import ProcessingContext
from dlkit.runtime.pipelines.pipeline import PrecisionValidationStep


class FakeInvoker:
    def __init__(self, dtype: torch.dtype):
        self.model = SimpleNamespace(dtype=dtype)


def test_precision_validation_step_raises_on_mismatch() -> None:
    step = PrecisionValidationStep(FakeInvoker(torch.float64))
    context = ProcessingContext(features={"x": torch.ones(2, dtype=torch.float32)})

    with pytest.raises(RuntimeError, match="expected torch.float64"):
        step.process(context)


def test_precision_validation_step_passes_when_aligned() -> None:
    step = PrecisionValidationStep(FakeInvoker(torch.float64))
    context = ProcessingContext(features={"x": torch.ones(2, dtype=torch.float64)})

    result = step.process(context)
    assert result is context, "PrecisionValidationStep should return the original context"
