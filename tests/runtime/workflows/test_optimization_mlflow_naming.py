"""Regression tests for MLflow naming inside optimization workflows."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import pytest

from dlkit.runtime.workflows.optimization.domain.models import (
    OptimizationDirection,
    Study,
)
from dlkit.runtime.workflows.optimization.infrastructure.tracking import (
    MLflowTrackingAdapter,
)
from dlkit.runtime.workflows.strategies.tracking import determine_study_name
from dlkit.tools.config import GeneralSettings


def _make_settings(run_name: str | None = None) -> GeneralSettings:
    mlflow = SimpleNamespace(run_name=run_name) if run_name is not None else None
    return cast("GeneralSettings", SimpleNamespace(SESSION=None, MLFLOW=mlflow))


def test_determine_study_name_prefers_explicit_run_name() -> None:
    settings = _make_settings(run_name="  explicit_name  ")
    optuna_config = SimpleNamespace(study_name=None)

    assert determine_study_name(settings, optuna_config) == "explicit_name"


def test_determine_study_name_uses_optuna_name_when_present() -> None:
    settings = _make_settings(run_name=None)
    optuna_config = SimpleNamespace(study_name="optuna_study")

    assert determine_study_name(settings, optuna_config) == "optuna_study"


def test_determine_study_name_generates_mlflow_style_random_name() -> None:
    settings = _make_settings(run_name=None)
    optuna_config = SimpleNamespace(study_name=None)

    with patch(
        "dlkit.runtime.workflows.strategies.tracking.naming.import_module",
        return_value=SimpleNamespace(_generate_random_name=lambda: "lucky-duck-42"),
    ) as import_module:
        assert determine_study_name(settings, optuna_config) == "lucky-duck-42"
        import_module.assert_called_once_with("mlflow.utils.name_utils")


def test_determine_study_name_fallback_without_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(run_name=None)
    optuna_config = SimpleNamespace(study_name=None)

    def raise_import(name: str, package: str | None = None):  # pragma: no cover - deterministic
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "dlkit.runtime.workflows.strategies.tracking.naming.import_module",
        raise_import,
    )

    fallback = determine_study_name(settings, optuna_config)
    assert fallback.startswith("dlkit-run-")


def test_adapter_parent_run_defaults_to_study_name_when_no_explicit_run_name() -> None:
    adapter = MLflowTrackingAdapter(
        mlflow_tracker=Mock(),
        mlflow_settings=None,
        session_name="experiment",
    )
    study = Study(
        study_id="study-1",
        study_name="random-name",
        direction=OptimizationDirection.MINIMIZE,
    )

    assert adapter._get_run_name_from_study(study) == "random-name"


def test_adapter_parent_run_respects_explicit_run_name_over_study() -> None:
    adapter = MLflowTrackingAdapter(
        mlflow_tracker=Mock(),
        mlflow_settings=SimpleNamespace(run_name="configured"),
        session_name="experiment",
    )
    study = Study(
        study_id="study-2",
        study_name="random-name",
        direction=OptimizationDirection.MINIMIZE,
    )

    assert adapter._get_run_name_from_study(study) == "configured"
