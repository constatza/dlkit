"""Tests for DataEntry.model_input field.

Covers the model_input bool field on DataEntry:
- True (default): entry included as positional model input
- False: entry excluded from model dispatch
- Non-bool values are rejected by Pydantic
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from dlkit.infrastructure.config.data_entries import Feature


class TestModelInputField:
    """Tests for the model_input bool field on DataEntry subclasses."""

    def test_model_input_true_accepted(self) -> None:
        """True is accepted (include as model input)."""
        feat = Feature("x", value=torch.zeros(4, 3), model_input=True)
        assert feat.model_input is True

    def test_model_input_false_accepted(self) -> None:
        """False is accepted (exclude from model call)."""
        feat = Feature("x", value=torch.zeros(4, 3), model_input=False)
        assert feat.model_input is False

    def test_model_input_default_is_true(self) -> None:
        """Default model_input is True (include as model input)."""
        feat = Feature("x", value=torch.zeros(4, 3))
        assert feat.model_input is True

    def test_model_input_str_raises(self) -> None:
        """String model_input raises ValidationError (hard-cut to bool)."""
        with pytest.raises(ValidationError):
            Feature("x", value=torch.zeros(4, 3), model_input="hidden")  # type: ignore[arg-type]

    def test_model_input_int_raises(self) -> None:
        """Integer model_input raises ValidationError (hard-cut to bool)."""
        with pytest.raises(ValidationError):
            Feature("x", value=torch.zeros(4, 3), model_input=0)  # type: ignore[arg-type]

    def test_model_input_none_raises(self) -> None:
        """None model_input raises ValidationError (hard-cut to bool)."""
        with pytest.raises(ValidationError):
            Feature("x", value=torch.zeros(4, 3), model_input=None)  # type: ignore[arg-type]
