"""Tests for resolve_factory_kwargs: single source of truth for factory-directive fields.

Regression coverage for the bug where `normalize` was silently dropped from
checkpoints because its re-injection logic was duplicated (and drifted out of
sync) across model construction, checkpoint serialization, and checkpoint
reconstruction.
"""

from __future__ import annotations

import pytest

from dlkit.engine.adapters.lightning.factory_kwargs import resolve_factory_kwargs
from dlkit.infrastructure.config.model_components import ModelComponentSettings


class _AcceptsNormalize:
    def __init__(self, normalize: str | None = None, activation: str | None = None) -> None:
        self.normalize = normalize
        self.activation = activation


class _RejectsNormalize:
    def __init__(self, activation: str | None = None) -> None:
        self.activation = activation


@pytest.fixture
def normalize_settings() -> ModelComponentSettings:
    """Settings with an explicit non-default normalize value."""
    return ModelComponentSettings.model_validate({"name": "Dummy", "normalize": "layer"})


@pytest.fixture
def unset_normalize_settings() -> ModelComponentSettings:
    """Settings with normalize left at its default (None)."""
    return ModelComponentSettings.model_validate({"name": "Dummy"})


def test_injects_directive_when_model_accepts_it(normalize_settings):
    """An explicit non-default value is added when the model's __init__ has that param."""
    resolved = resolve_factory_kwargs(normalize_settings, _AcceptsNormalize, {})
    assert resolved["normalize"] == "layer"


def test_omits_directive_when_model_rejects_it(normalize_settings):
    """No injection when the model class's __init__ has no matching parameter."""
    resolved = resolve_factory_kwargs(normalize_settings, _RejectsNormalize, {})
    assert "normalize" not in resolved


def test_omits_directive_when_model_cls_is_none(normalize_settings):
    """Unresolved model class (e.g. import failed) leaves kwargs untouched."""
    resolved = resolve_factory_kwargs(normalize_settings, None, {"in_features": 4})
    assert resolved == {"in_features": 4}


def test_omits_directive_when_unset(unset_normalize_settings):
    """A field left at its pydantic default (None) is never injected."""
    resolved = resolve_factory_kwargs(unset_normalize_settings, _AcceptsNormalize, {})
    assert "normalize" not in resolved


def test_does_not_overwrite_existing_kwarg(normalize_settings):
    """An explicitly-provided raw kwarg wins over the settings-derived value."""
    resolved = resolve_factory_kwargs(normalize_settings, _AcceptsNormalize, {"normalize": "batch"})
    assert resolved["normalize"] == "batch"


def test_does_not_mutate_input_kwargs(normalize_settings):
    """The input kwargs dict is copied, not mutated in place."""
    original = {"in_features": 4}
    resolve_factory_kwargs(normalize_settings, _AcceptsNormalize, original)
    assert original == {"in_features": 4}
