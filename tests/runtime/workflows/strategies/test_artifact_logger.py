"""Tests for ArtifactLogger model registration behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from torch import Tensor

from dlkit.core.models.wrappers.base import ProcessingLightningWrapper
from dlkit.runtime.workflows.factories.build_factory import BuildComponents
from dlkit.runtime.workflows.strategies.tracking.artifact_logger import (
    ArtifactLogger,
    TAG_LOGGED_MODEL_URI,
    TAG_LOGGED_MODEL_ARTIFACT_PATH,
    TAG_MODEL_CLASS,
    TAG_MODEL_REGISTRATION_ENABLED,
    _resolve_model_class_name,
)
from dlkit.runtime.workflows.strategies.tracking.interfaces import IRunContext
from dlkit.tools.config.general_settings import GeneralSettings
from dlkit.tools.config.mlflow_settings import MLflowSettings


# ---------------------------------------------------------------------------
# Minimal test doubles for wrapper unwrapping tests
# ---------------------------------------------------------------------------


class _InnerNet:
    """Stub inner model with a distinctive class name."""


class _ConcreteWrapper(ProcessingLightningWrapper):
    """Minimal concrete wrapper used only to satisfy isinstance checks."""

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:  # type: ignore[override]
        raise NotImplementedError

    def _run_step(self, batch: Any, batch_idx: int, stage: str) -> tuple[Tensor, int | None, Any]:
        raise NotImplementedError


@pytest.fixture
def wrapped_model() -> _ConcreteWrapper:
    """A ProcessingLightningWrapper instance whose inner model is _InnerNet.

    Uses object.__new__ to bypass Lightning's __init__ — we only need the
    isinstance relationship and the .model attribute.
    """
    wrapper: _ConcreteWrapper = object.__new__(_ConcreteWrapper)
    wrapper.model = _InnerNet()
    return wrapper


class _RecordingRunContext(IRunContext):
    """Recording run context for registration assertions."""

    def __init__(self):
        self._run_id = "test-run"
        self.logged_model_calls: list[dict[str, Any]] = []
        self.alias_calls: list[tuple[str, str, int]] = []
        self.model_version_tag_calls: list[tuple[str, int, str, str]] = []
        self.tags: dict[str, str] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_text(self, text: str, artifact_file: str) -> None:
        pass

    def log_artifact(self, artifact_path: Path, artifact_dir: str = "") -> None:
        pass

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_dataset(
        self,
        dataset: Any,
        context: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        pass

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        *,
        registered_model_name: str | None = None,
        signature: Any | None = None,
        input_example: Any | None = None,
    ) -> str | None:
        self.logged_model_calls.append({
            "artifact_path": artifact_path,
            "registered_model_name": registered_model_name,
        })
        return f"runs:/{self._run_id}/{artifact_path}"

    def get_latest_model_version(
        self,
        model_name: str,
        *,
        run_id: str | None = None,
        artifact_path: str | None = None,
    ) -> int | None:
        assert run_id == self._run_id
        assert artifact_path in (None, "model")
        return 7

    def set_model_alias(self, model_name: str, alias: str, version: int) -> None:
        self.alias_calls.append((model_name, alias, version))

    def set_model_version_tag(
        self,
        model_name: str,
        version: int,
        key: str,
        value: str,
    ) -> None:
        self.model_version_tag_calls.append((model_name, version, key, value))


def _build_components(model: Any) -> BuildComponents:
    return BuildComponents(
        model=model,
        datamodule=Mock(),
        trainer=None,
        shape_spec=None,
        meta={},
    )


def test_registers_model_with_class_name_without_default_aliases_or_tags() -> None:
    @dataclass(frozen=True, slots=True)
    class FancyNet:
        pass

    settings = GeneralSettings(
        MODEL=None,
        MLFLOW=MLflowSettings(
            register_model=True,
        ),
    )

    logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()
    components = _build_components(FancyNet())

    pending = logger.maybe_register_model(settings, components, run_context)
    logger.finalize_model_registration(pending, run_context)

    assert run_context.logged_model_calls[0]["registered_model_name"] == "FancyNet"
    assert run_context.alias_calls == []
    assert run_context.model_version_tag_calls == []
    assert run_context.tags["mlflow_registered_model_name"] == "FancyNet"
    assert run_context.tags["mlflow_registered_model_version"] == "7"
    assert run_context.tags[TAG_LOGGED_MODEL_URI] == "runs:/test-run/model"
    assert run_context.tags[TAG_LOGGED_MODEL_ARTIFACT_PATH] == "model"
    assert run_context.tags[TAG_MODEL_CLASS] == "FancyNet"
    assert run_context.tags[TAG_MODEL_REGISTRATION_ENABLED] == "true"


def test_logs_model_without_registration_when_disabled() -> None:
    @dataclass(frozen=True, slots=True)
    class FancyNet:
        pass

    settings = GeneralSettings(
        MLFLOW=MLflowSettings(
            register_model=False,
        ),
    )
    logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()

    pending = logger.maybe_register_model(settings, _build_components(FancyNet()), run_context)
    logger.finalize_model_registration(pending, run_context)

    assert len(run_context.logged_model_calls) == 1
    assert run_context.logged_model_calls[0]["registered_model_name"] is None
    assert run_context.alias_calls == []
    assert run_context.model_version_tag_calls == []
    assert run_context.tags[TAG_LOGGED_MODEL_URI] == "runs:/test-run/model"
    assert run_context.tags[TAG_LOGGED_MODEL_ARTIFACT_PATH] == "model"
    assert run_context.tags[TAG_MODEL_CLASS] == "FancyNet"
    assert run_context.tags[TAG_MODEL_REGISTRATION_ENABLED] == "false"


def test_merges_configured_aliases_and_model_version_tags() -> None:
    @dataclass(frozen=True, slots=True)
    class FancyNet:
        pass

    settings = GeneralSettings(
        MLFLOW=MLflowSettings(
            register_model=True,
            registered_model_aliases=("benchmark_high_precision", "dataset_A_latest"),
            registered_model_version_tags={"team": "platform"},
        ),
    )
    logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()

    pending = logger.maybe_register_model(settings, _build_components(FancyNet()), run_context)
    logger.finalize_model_registration(pending, run_context)

    aliases = sorted(alias for _, alias, _ in run_context.alias_calls)
    assert aliases == ["benchmark_high_precision", "dataset_A_latest"]
    assert ("FancyNet", 7, "team", "platform") in run_context.model_version_tag_calls


# ---------------------------------------------------------------------------
# _resolve_model_class_name
# ---------------------------------------------------------------------------


def test_resolve_model_class_name_returns_plain_model_class() -> None:
    class _PlainModel:
        pass

    assert _resolve_model_class_name(_PlainModel()) == "_PlainModel"


def test_resolve_model_class_name_unwraps_processing_lightning_wrapper(
    wrapped_model: _ConcreteWrapper,
) -> None:
    assert _resolve_model_class_name(wrapped_model) == "_InnerNet"


def test_maybe_register_model_uses_inner_model_class_name_for_tag(
    wrapped_model: _ConcreteWrapper,
) -> None:
    settings = GeneralSettings(
        MLFLOW=MLflowSettings(
            register_model=False,
        ),
    )
    artifact_logger = ArtifactLogger(tracker=Mock())
    run_context = _RecordingRunContext()

    artifact_logger.maybe_register_model(settings, _build_components(wrapped_model), run_context)

    assert run_context.tags[TAG_MODEL_CLASS] == "_InnerNet"
