"""Tests for SettingsLogger.log_model_parameters.

Verifies that logged params come from the model's already-populated
``.hparams`` (set via Lightning's save_hyperparameters() at construction
time), not re-derived from settings at logging time.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import pytest

from dlkit.common.hooks import ParamValue
from dlkit.engine.tracking.settings_logger import SettingsLogger
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.model_components import ModelComponentSettings
from dlkit.infrastructure.config.run_settings import RunSettings


class FakeRunContext:
    """Minimal IRunContext double capturing logged params."""

    def __init__(self) -> None:
        self.logged_params: dict[str, ParamValue] = {}
        self.call_count = 0

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        self.call_count += 1
        self.logged_params.update(dict(params))


@pytest.fixture
def logger() -> SettingsLogger:
    return SettingsLogger()


@pytest.fixture
def run_context() -> FakeRunContext:
    return FakeRunContext()


@pytest.fixture
def settings_with_model() -> JobConfig:
    return JobConfig(run=RunSettings(type="train"), model=ModelComponentSettings(name="FFNN"))


@pytest.fixture
def settings_without_model() -> JobConfig:
    return JobConfig(run=RunSettings(type="train"), model=None)


class TestLogModelParameters:
    def test_logs_model_hparams_verbatim(
        self, logger: SettingsLogger, run_context: FakeRunContext, settings_with_model: JobConfig
    ) -> None:
        model = SimpleNamespace(hparams={"activation": "gelu", "num_layers": 2})
        logger.log_model_parameters(model, run_context, settings_with_model)
        assert run_context.logged_params == {"activation": "gelu", "num_layers": 2}
        assert run_context.call_count == 1

    def test_skips_when_settings_model_is_none(
        self, logger: SettingsLogger, run_context: FakeRunContext, settings_without_model: JobConfig
    ) -> None:
        model = SimpleNamespace(hparams={"activation": "gelu"})
        logger.log_model_parameters(model, run_context, settings_without_model)
        assert run_context.call_count == 0

    def test_skips_when_hparams_empty(
        self, logger: SettingsLogger, run_context: FakeRunContext, settings_with_model: JobConfig
    ) -> None:
        model = SimpleNamespace(hparams={})
        logger.log_model_parameters(model, run_context, settings_with_model)
        assert run_context.call_count == 0

    def test_wraps_failures_in_runtime_error(
        self, logger: SettingsLogger, run_context: FakeRunContext, settings_with_model: JobConfig
    ) -> None:
        class BoomModel:
            @property
            def hparams(self) -> dict[str, ParamValue]:
                raise ValueError("boom")

        with pytest.raises(RuntimeError, match="Couldn't log model parameters"):
            logger.log_model_parameters(BoomModel(), run_context, settings_with_model)
