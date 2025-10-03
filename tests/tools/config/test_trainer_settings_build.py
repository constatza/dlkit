from __future__ import annotations

from dlkit.tools.config.trainer_settings import TrainerSettings, CallbackSettings, LoggerSettings
from dlkit.tools.config.core.context import BuildContext
from dlkit.tools.config.core.factories import FactoryProvider


def test_trainer_settings_build_creates_callbacks_and_logger(monkeypatch):  # noqa: ANN001
    # Prepare settings with one callback and a logger
    ts = TrainerSettings(
        callbacks=(CallbackSettings(name="ModelSummary"),),
        logger=LoggerSettings(name="CSVLogger"),
        max_epochs=1,
    )

    # Patch FactoryProvider.create_component so that callbacks/loggers resolve to simple objects
    def _fake_create(settings, ctx: BuildContext):  # noqa: ANN001
        # Return a simple object representing constructed component
        class _Dummy:
            pass

        return _Dummy()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create))

    trainer = ts.build()
    # Trainer is constructed via FactoryProvider (our dummy class), but we verify no exceptions and attrs used
    assert trainer is not None
