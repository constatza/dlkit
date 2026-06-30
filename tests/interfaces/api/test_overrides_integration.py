"""Integration tests for request-level runtime overrides."""

from __future__ import annotations

import pytest

from dlkit.engine.workflows.entrypoints._overrides import (
    apply_runtime_overrides,
    build_runtime_overrides,
)
from dlkit.infrastructure.config.data_settings import DataSettings
from dlkit.infrastructure.config.job_config import TrainingJobConfig
from dlkit.infrastructure.config.model_components import ModelComponentSettings
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    LBFGSSettings,
    MuonSettings,
)
from dlkit.infrastructure.config.run_settings import RunSettings
from dlkit.infrastructure.config.trainer_settings import TrainerSettings
from dlkit.infrastructure.config.training_settings import TrainingSettings


def _require_training(job: TrainingJobConfig) -> TrainingSettings:
    """Extract required training settings from a TrainingJobConfig.

    Args:
        job: A training job config.

    Returns:
        The training settings section.
    """
    training = job.training
    assert training is not None
    return training


def _require_data(job: TrainingJobConfig) -> DataSettings:
    """Extract required data settings from a TrainingJobConfig.

    Args:
        job: A training job config.

    Returns:
        The data settings section.
    """
    data = job.data
    assert data is not None
    return data


def _require_numeric_lr(settings: TrainingSettings) -> int | float:
    """Extract learning rate from training settings.

    Args:
        settings: Training settings with optimizer configuration.

    Returns:
        The numeric learning rate value.
    """
    optimizer = settings.optimizer.default_optimizer
    assert isinstance(optimizer, AdamWSettings | AdamSettings | LBFGSSettings | MuonSettings)
    lr = optimizer.lr
    assert isinstance(lr, int | float)
    return lr


@pytest.fixture
def sample_job() -> TrainingJobConfig:
    """Minimal TrainingJobConfig with training and data sections for override testing.

    Returns:
        TrainingJobConfig with max_epochs=50 and batch_size=16.
    """
    return TrainingJobConfig(
        run=RunSettings(type="train"),
        model=ModelComponentSettings.model_validate({"class": "DummyModel"}),
        data=DataSettings(batch_size=16),
        training=TrainingSettings(trainer=TrainerSettings(max_epochs=50)),
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
        self, sample_job: TrainingJobConfig
    ) -> None:
        overrides = build_runtime_overrides(additional_overrides={})
        assert overrides == {}

        result = apply_runtime_overrides(sample_job, **overrides)
        result_training = _require_training(result)
        sample_training = _require_training(sample_job)
        result_data = _require_data(result)
        sample_data = _require_data(sample_job)

        assert result_training.trainer.max_epochs == sample_training.trainer.max_epochs
        assert result_data.batch_size == sample_data.batch_size

    def test_overrides_applied_when_values_provided(self, sample_job: TrainingJobConfig) -> None:
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

        result = apply_runtime_overrides(sample_job, **overrides)
        result_training = _require_training(result)
        result_data = _require_data(result)

        assert result_training.trainer.max_epochs == 100
        assert result_data.batch_size == 64
        assert float(_require_numeric_lr(result_training)) == pytest.approx(0.01)

    def test_partial_overrides_preserve_non_overridden_values(
        self, sample_job: TrainingJobConfig
    ) -> None:
        overrides = build_runtime_overrides(epochs=200, additional_overrides={})
        assert overrides == {"epochs": 200}

        result = apply_runtime_overrides(sample_job, **overrides)
        result_training = _require_training(result)
        result_data = _require_data(result)
        sample_data = _require_data(sample_job)

        assert result_training.trainer.max_epochs == 200
        assert result_data.batch_size == sample_data.batch_size

    def test_none_values_explicitly_ignored(self, sample_job: TrainingJobConfig) -> None:
        overrides = build_runtime_overrides(
            epochs=None,
            batch_size=None,
            learning_rate=50,
            additional_overrides={},
        )
        assert overrides == {"learning_rate": 50}

        result = apply_runtime_overrides(sample_job, **overrides)
        result_training = _require_training(result)
        sample_training = _require_training(sample_job)
        result_data = _require_data(result)
        sample_data = _require_data(sample_job)

        assert result_training.trainer.max_epochs == sample_training.trainer.max_epochs
        assert result_data.batch_size == sample_data.batch_size
        assert float(_require_numeric_lr(result_training)) == pytest.approx(50)


def test_override_workflow_summary() -> None:
    overrides = build_runtime_overrides(
        epochs=None,
        batch_size=None,
        learning_rate=None,
        additional_overrides={},
    )
    assert overrides == {}
