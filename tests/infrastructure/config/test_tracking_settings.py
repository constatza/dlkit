"""Tests for tracking settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlkit.infrastructure.config.tracking_settings import TrackingSettings


def test_tracking_settings_default_model_serialization_format_is_pickle() -> None:
    settings = TrackingSettings()

    assert settings.model_serialization_format == "pickle"


def test_tracking_settings_default_max_retries_matches_process_wide_default() -> None:
    settings = TrackingSettings()

    assert settings.max_retries == 5


def test_tracking_settings_accepts_explicit_max_retries() -> None:
    settings = TrackingSettings(max_retries=10)

    assert settings.max_retries == 10


def test_tracking_settings_accepts_pt2_model_serialization_format() -> None:
    settings = TrackingSettings(model_serialization_format="pt2")

    assert settings.model_serialization_format == "pt2"


def test_tracking_settings_rejects_unknown_model_serialization_format() -> None:
    with pytest.raises(ValidationError):
        TrackingSettings.model_validate({"model_serialization_format": "onnx"})


def test_tracking_settings_default_on_connectivity_failure_is_raise() -> None:
    """Fail-fast is the safe default: an unattended run can't ask a human, and
    silently falling back risks the run landing somewhere the user doesn't expect.
    """
    settings = TrackingSettings()

    assert settings.on_connectivity_failure == "raise"


def test_tracking_settings_accepts_fallback_local_on_connectivity_failure() -> None:
    settings = TrackingSettings(on_connectivity_failure="fallback_local")

    assert settings.on_connectivity_failure == "fallback_local"


def test_tracking_settings_rejects_unknown_on_connectivity_failure() -> None:
    with pytest.raises(ValidationError):
        TrackingSettings.model_validate({"on_connectivity_failure": "retry"})
