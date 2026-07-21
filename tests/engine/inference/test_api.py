"""Tests for `load_model_from_settings`'s seeding and precision chokepoint behavior."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from dlkit.engine.inference.api import load_model_from_settings
from dlkit.infrastructure.config.job_config import InferenceJobConfig
from dlkit.infrastructure.precision.strategy import PrecisionStrategy


def test_load_model_from_settings_applies_resolved_seed(
    make_inference_settings: Callable[..., InferenceJobConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured run seed must be applied via `apply_global_seed`."""
    settings = make_inference_settings(seed=123)
    spy = MagicMock()
    monkeypatch.setattr("dlkit.engine.inference.api.apply_global_seed", spy)

    load_model_from_settings(settings, device="cpu")

    spy.assert_called_once_with(123, workers=True)


def test_load_model_from_settings_defaults_seed_when_unset(
    make_inference_settings: Callable[..., InferenceJobConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset run seed must resolve to `RunSettings.resolve_seed`'s default."""
    settings = make_inference_settings()
    spy = MagicMock()
    monkeypatch.setattr("dlkit.engine.inference.api.apply_global_seed", spy)

    load_model_from_settings(settings, device="cpu")

    spy.assert_called_once_with(settings.run.resolve_seed(), workers=True)


def test_load_model_from_settings_seeds_global_rng_reproducibly(
    make_inference_settings: Callable[..., InferenceJobConfig],
) -> None:
    """Two loads with the same seed must leave the global RNG in the same state.

    Exercises the real `apply_global_seed` (no mocking) to prove seeding has an
    observable effect: a `torch.rand` draw immediately after loading must be
    identical across two independent calls sharing the same configured seed.
    """
    settings = make_inference_settings(seed=7)

    load_model_from_settings(settings, device="cpu")
    first_draw = torch.rand(1)

    load_model_from_settings(settings, device="cpu")
    second_draw = torch.rand(1)

    assert torch.equal(first_draw, second_draw)


def test_load_model_from_settings_resolves_precision_from_run_when_not_overridden(
    make_inference_settings: Callable[..., InferenceJobConfig],
) -> None:
    """`settings.run.precision` must populate the predictor config when unset by the caller."""
    settings = make_inference_settings(precision="64")

    predictor = load_model_from_settings(settings, device="cpu")

    assert predictor._config.precision == PrecisionStrategy.FULL_64


def test_load_model_from_settings_explicit_precision_overrides_run_precision(
    make_inference_settings: Callable[..., InferenceJobConfig],
) -> None:
    """An explicit `precision=` kwarg must win over `settings.run.precision`."""
    settings = make_inference_settings(precision="64")

    predictor = load_model_from_settings(
        settings, device="cpu", precision=PrecisionStrategy.FULL_32
    )

    assert predictor._config.precision == PrecisionStrategy.FULL_32


def test_load_model_from_settings_precision_none_when_run_precision_unset(
    make_inference_settings: Callable[..., InferenceJobConfig],
) -> None:
    """With no run precision and no explicit override, the predictor falls back to inference."""
    settings = make_inference_settings()

    predictor = load_model_from_settings(settings, device="cpu")

    assert predictor._config.precision is None
