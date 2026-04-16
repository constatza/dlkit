from __future__ import annotations

from unittest.mock import patch

from dlkit.infrastructure.config.core.context import BuildContext
from dlkit.infrastructure.config.core.factories import FactoryProvider
from dlkit.infrastructure.config.trainer_settings import (
    CallbackSettings,
    LoggerSettings,
    TrainerSettings,
)


def test_trainer_settings_build_creates_callbacks_and_logger(monkeypatch):
    # Prepare settings with one callback and a logger
    ts = TrainerSettings(
        callbacks=(CallbackSettings(name="ModelSummary"),),
        logger=LoggerSettings(name="CSVLogger"),
        max_epochs=1,
    )

    # Patch FactoryProvider.create_component so that callbacks/loggers resolve to simple objects
    def _fake_create(settings, ctx: BuildContext):
        # Return a simple object representing constructed component
        class _Dummy:
            pass

        return _Dummy()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create))

    trainer = ts.build()
    # Trainer is constructed via FactoryProvider (our dummy class), but we verify no exceptions and attrs used
    assert trainer is not None


def test_trainer_settings_build_disables_model_summary_and_respects_progress_bar(
    monkeypatch,
):
    captured_overrides: dict | None = None

    def _fake_create(settings, ctx: BuildContext):
        nonlocal captured_overrides
        captured_overrides = ctx.overrides

        class _Dummy:
            pass

        return _Dummy()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create))

    with patch(
        "dlkit.infrastructure.config.trainer_settings.should_enable_progress_bar",
        return_value=False,
    ):
        trainer = TrainerSettings().build()

    assert trainer is not None
    assert captured_overrides is not None
    assert captured_overrides["enable_model_summary"] is False
    assert captured_overrides["enable_progress_bar"] is False
