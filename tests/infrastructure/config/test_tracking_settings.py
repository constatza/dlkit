"""Tests for tracking settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlkit.infrastructure.config.tracking_settings import TrackingSettings


def test_tracking_settings_default_model_serialization_format_is_pickle() -> None:
    settings = TrackingSettings()

    assert settings.model_serialization_format == "pickle"


def test_tracking_settings_accepts_pt2_model_serialization_format() -> None:
    settings = TrackingSettings(model_serialization_format="pt2")

    assert settings.model_serialization_format == "pt2"


def test_tracking_settings_rejects_unknown_model_serialization_format() -> None:
    with pytest.raises(ValidationError):
        TrackingSettings.model_validate({"model_serialization_format": "onnx"})
