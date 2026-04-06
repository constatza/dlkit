"""Tests for DataEntry.model_input field validation.

Covers the model_input field on DataEntry:
- str validator: empty string rejected, invalid identifiers rejected
- digit strings and identifiers accepted
- int values accepted directly (positional index)
- bool values accepted (True/False semantics)
- None accepted (exclude from model call)
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from dlkit.infrastructure.config.data_entries import Feature


class TestModelInputValidator:
    """Tests for the model_input field validator on DataEntry subclasses."""

    def test_model_input_empty_str_raises(self) -> None:
        """Empty string model_input raises ValidationError."""
        with pytest.raises(ValidationError, match="non-empty"):
            Feature("x", value=torch.zeros(4, 3), model_input="")

    def test_model_input_invalid_str_raises(self) -> None:
        """Non-identifier, non-digit string raises ValidationError."""
        with pytest.raises(ValidationError, match="digit string.*identifier"):
            Feature("x", value=torch.zeros(4, 3), model_input="not-valid!")

    def test_model_input_hyphen_str_raises(self) -> None:
        """Hyphenated string is not a valid identifier."""
        with pytest.raises(ValidationError):
            Feature("x", value=torch.zeros(4, 3), model_input="my-kwarg")

    def test_model_input_digit_str_valid(self) -> None:
        """Digit string is accepted as positional index."""
        feat = Feature("x", value=torch.zeros(4, 3), model_input="0")
        assert feat.model_input == "0"

    def test_model_input_multidigit_str_valid(self) -> None:
        """Multi-digit string (e.g. '10') is accepted."""
        feat = Feature("x", value=torch.zeros(4, 3), model_input="10")
        assert feat.model_input == "10"

    def test_model_input_kwarg_name_valid(self) -> None:
        """Valid Python identifier string is accepted as kwarg name."""
        feat = Feature("x", value=torch.zeros(4, 3), model_input="hidden")
        assert feat.model_input == "hidden"

    def test_model_input_underscore_identifier_valid(self) -> None:
        """Underscore-prefixed identifier is a valid Python identifier."""
        feat = Feature("x", value=torch.zeros(4, 3), model_input="_hidden")
        assert feat.model_input == "_hidden"

    def test_model_input_int_accepted_directly(self) -> None:
        """Integer model_input is accepted directly (no coercion needed)."""
        feat = Feature("x", value=torch.zeros(4, 3), model_input=0)
        assert feat.model_input == 0

    def test_model_input_int_1_accepted(self) -> None:
        """Integer 1 is accepted as positional index 1."""
        feat = Feature("x", value=torch.zeros(4, 3), model_input=1)
        assert feat.model_input == 1

    def test_model_input_true_accepted(self) -> None:
        """True is accepted (kwarg dispatch by entry name)."""
        feat = Feature("x", value=torch.zeros(4, 3), model_input=True)
        assert feat.model_input is True

    def test_model_input_false_accepted(self) -> None:
        """False is accepted (exclude from model call)."""
        feat = Feature("x", value=torch.zeros(4, 3), model_input=False)
        assert feat.model_input is False

    def test_model_input_none_accepted(self) -> None:
        """None is accepted (exclude from model call)."""
        feat = Feature("x", value=torch.zeros(4, 3), model_input=None)
        assert feat.model_input is None

    def test_model_input_default_is_true(self) -> None:
        """Default model_input is True (kwarg by entry name)."""
        feat = Feature("x", value=torch.zeros(4, 3))
        assert feat.model_input is True
