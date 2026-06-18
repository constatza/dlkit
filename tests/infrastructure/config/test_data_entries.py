"""Tests for DataEntry fields: model_input.

Covers:
- model_input bool field: True/False/non-bool rejection
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
from pydantic import ValidationError

from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.entry_types import ValueEntry


class TestModelInputField:
    """Tests for the model_input bool field on DataEntry subclasses."""

    def test_model_input_true_accepted(self) -> None:
        """True is accepted (include as model input)."""
        feat = ValueEntry(
            name="x", value=torch.zeros(4, 3), data_role=DataRole.FEATURE, model_input=True
        )
        assert feat.model_input is True

    def test_model_input_false_accepted(self) -> None:
        """False is accepted (exclude from model call)."""
        feat = ValueEntry(
            name="x", value=torch.zeros(4, 3), data_role=DataRole.FEATURE, model_input=False
        )
        assert feat.model_input is False

    def test_model_input_default_is_true(self) -> None:
        """Default model_input is True (include as model input)."""
        feat = ValueEntry(name="x", value=torch.zeros(4, 3), data_role=DataRole.FEATURE)
        assert feat.model_input is True

    def test_model_input_str_raises(self) -> None:
        """String model_input raises ValidationError (hard-cut to bool)."""
        with pytest.raises(ValidationError):
            ValueEntry(
                name="x",
                value=torch.zeros(4, 3),
                data_role=DataRole.FEATURE,
                model_input=cast("bool", "hidden"),
            )

    def test_model_input_int_raises(self) -> None:
        """Integer model_input raises ValidationError (hard-cut to bool)."""
        with pytest.raises(ValidationError):
            ValueEntry(
                name="x",
                value=torch.zeros(4, 3),
                data_role=DataRole.FEATURE,
                model_input=cast("bool", 0),
            )

    def test_model_input_none_raises(self) -> None:
        """None model_input raises ValidationError (hard-cut to bool)."""
        with pytest.raises(ValidationError):
            ValueEntry(
                name="x",
                value=torch.zeros(4, 3),
                data_role=DataRole.FEATURE,
                model_input=cast("bool", None),
            )

    def test_model_input_int_one_raises(self) -> None:
        """Integer 1 raises ValidationError (strict bool — no coercion from int)."""
        with pytest.raises(ValidationError):
            ValueEntry(
                name="x",
                value=torch.zeros(4, 3),
                data_role=DataRole.FEATURE,
                model_input=cast("bool", 1),
            )

    def test_model_input_kwarg_str_raises(self) -> None:
        """String 'kwarg' raises ValidationError (strict bool — no strings accepted)."""
        with pytest.raises(ValidationError):
            ValueEntry(
                name="x",
                value=torch.zeros(4, 3),
                data_role=DataRole.FEATURE,
                model_input=cast("bool", "kwarg"),
            )
