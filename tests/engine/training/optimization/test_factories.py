"""Tests for optimizer and scheduler factories."""

from __future__ import annotations

import pytest
import torch.optim as toptim

from dlkit.engine.training.optimization.factories import _import_optimizer_class


def test_import_by_name_and_module() -> None:
    """Import a class by name when fallback module is provided.

    Should resolve "Adam" to torch.optim.Adam when given fallback_module.
    """
    cls = _import_optimizer_class("Adam", "torch.optim")
    assert cls is toptim.Adam


def test_import_with_colon_path() -> None:
    """Import a class using explicit module:class notation.

    Should resolve "torch.optim:AdamW" to torch.optim.AdamW.
    """
    cls = _import_optimizer_class("torch.optim:AdamW", "unused_fallback")
    assert cls is toptim.AdamW


def test_import_raises_on_bad_module() -> None:
    """Import raises when module cannot be found.

    Should raise ModuleNotFoundError or ImportError for nonexistent modules.
    """
    with pytest.raises((ModuleNotFoundError, ImportError)):
        _import_optimizer_class("NonExistent", "nonexistent.module.xyz")
