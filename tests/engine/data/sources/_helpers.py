"""Pure assertion helpers for engine.data.sources tests."""

from __future__ import annotations

import torch


def assert_dtype(tensor: torch.Tensor, expected: torch.dtype) -> None:
    """Assert that a tensor has the expected dtype.

    Args:
        tensor: The tensor to check.
        expected: The expected ``torch.dtype``.

    Raises:
        AssertionError: When ``tensor.dtype != expected``.
    """
    assert tensor.dtype == expected, f"Expected dtype {expected}, got {tensor.dtype}"
