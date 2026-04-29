"""Integration tests for request-level runtime overrides."""

from __future__ import annotations

import pytest

from dlkit.engine.workflows.entrypoints._overrides import (
    apply_runtime_overrides,
    build_runtime_overrides,
)
from dlkit.infrastructure.config import GeneralSettings
from dlkit.infrastructure.config.dataloader_settings import DataloaderSettings
from dlkit.infrastructure.config.datamodule_settings import DataModuleSettings
from dlkit.infrastructure.config.session_settings import SessionSettings
from dlkit.infrastructure.config.training_settings import TrainingSettings


def _require_training(settings: GeneralSettings) -> TrainingSettings:
    training = settings.TRAINING
    assert training is not None
    return training


def _require_datamodule(settings: GeneralSettings) -> DataModuleSettings:
    datamodule = settings.DATAMODULE
    assert datamodule is not None
    return datamodule


def _require_numeric_lr(settings: TrainingSettings) -> int | float:
    lr = settings.optimizer.lr
    assert isinstance(lr, int | float)
    return lr


@pytest.fixture
def sample_settings() -> GeneralSettings:
    return GeneralSettings(
        SESSION=SessionSettings(workflow="train"),
        TRAINING=TrainingSettings(epochs=50),
        DATAMODULE=DataModuleSettings(dataloader=DataloaderSettings(batch_size=16)),
    )


class TestOverrideIntegration:
    """Integration tests for runtime override behavior."""

    def test_build_runtime_overrides_ignores_none_defaults(self) -> None:
        overrides = build_runtime_overrides(
            checkpoint_path=None,
            epochs=None,
            batch_size=None,
            learning_rate=None,
            additional_overrides={},
        )
        assert overrides == {}

    def test_settings_values_preserved_when_no_overrides(
        self, sample_settings: GeneralSettings
    ) -> None:
        overrides = build_runtime_overrides(additional_overrides={})
        assert overrides == {}

        result = apply_runtime_overrides(sample_settings, **overrides)
        result_training = _require_training(result)
        sample_training = _require_training(sample_settings)
        result_datamodule = _require_datamodule(result)
        sample_datamodule = _require_datamodule(sample_settings)

        assert result_training.epochs == sample_training.epochs
        assert result_datamodule.dataloader.batch_size == sample_datamodule.dataloader.batch_size

    def test_overrides_applied_when_values_provided(self, sample_settings: GeneralSettings) -> None:
        overrides = build_runtime_overrides(
            epochs=100,
            batch_size=64,
            learning_rate=0.01,
            additional_overrides={},
        )
        assert overrides == {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.01,
        }

        result = apply_runtime_overrides(sample_settings, **overrides)
        result_training = _require_training(result)
        result_datamodule = _require_datamodule(result)

        assert result_training.epochs == 100
        assert result_datamodule.dataloader.batch_size == 64
        assert float(_require_numeric_lr(result_training)) == pytest.approx(0.01)

    def test_partial_overrides_preserve_non_overridden_values(
        self, sample_settings: GeneralSettings
    ) -> None:
        overrides = build_runtime_overrides(epochs=200, additional_overrides={})
        assert overrides == {"epochs": 200}

        result = apply_runtime_overrides(sample_settings, **overrides)
        result_training = _require_training(result)
        result_datamodule = _require_datamodule(result)
        sample_datamodule = _require_datamodule(sample_settings)

        assert result_training.epochs == 200
        assert result_datamodule.dataloader.batch_size == sample_datamodule.dataloader.batch_size

    def test_none_values_explicitly_ignored(self, sample_settings: GeneralSettings) -> None:
        overrides = build_runtime_overrides(
            epochs=None,
            batch_size=None,
            learning_rate=50,
            additional_overrides={},
        )
        assert overrides == {"learning_rate": 50}

        result = apply_runtime_overrides(sample_settings, **overrides)
        result_training = _require_training(result)
        sample_training = _require_training(sample_settings)
        result_datamodule = _require_datamodule(result)
        sample_datamodule = _require_datamodule(sample_settings)

        assert result_training.epochs == sample_training.epochs
        assert result_datamodule.dataloader.batch_size == sample_datamodule.dataloader.batch_size
        assert float(_require_numeric_lr(result_training)) == pytest.approx(50)


def test_override_workflow_summary() -> None:
    overrides = build_runtime_overrides(
        epochs=None,
        batch_size=None,
        learning_rate=None,
        additional_overrides={},
    )
    assert overrides == {}
