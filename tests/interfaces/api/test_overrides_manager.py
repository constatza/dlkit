from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dlkit.engine.workflows.entrypoints._overrides import (
    apply_runtime_overrides,
    validate_runtime_overrides,
)
from dlkit.infrastructure.config.data_settings import DataSettings
from dlkit.infrastructure.config.experiment_settings import ExperimentSettings
from dlkit.infrastructure.config.job_config import TrainingJobConfig
from dlkit.infrastructure.config.model_components import ModelComponentSettings
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    LBFGSSettings,
    MuonSettings,
)
from dlkit.infrastructure.config.run_settings import RunSettings
from dlkit.infrastructure.config.training_settings import TrainingSettings
from dlkit.interfaces.api.domain.override_types import TrainingOverrides


def _base_settings() -> TrainingJobConfig:
    """Build a minimal TrainingJobConfig for override tests.

    Returns:
        TrainingJobConfig with all required sections populated with defaults.
    """
    return TrainingJobConfig(
        run=RunSettings(type="train"),
        model=ModelComponentSettings.model_validate({"class": "DummyModel"}),
        data=DataSettings(batch_size=64),
        training=TrainingSettings(),
        experiment=ExperimentSettings(name="test_experiment"),
    )


def _require_training(settings: TrainingJobConfig) -> TrainingSettings:
    """Extract required training settings.

    Args:
        settings: A training job config.

    Returns:
        TrainingSettings from the job config.
    """
    training = settings.training
    assert training is not None
    return training


def _require_data(settings: TrainingJobConfig) -> DataSettings:
    """Extract required data settings.

    Args:
        settings: A training job config.

    Returns:
        DataSettings from the job config.
    """
    data = settings.data
    assert data is not None
    return data


def _require_experiment(settings: TrainingJobConfig) -> ExperimentSettings:
    """Extract required experiment settings.

    Args:
        settings: A training job config.

    Returns:
        ExperimentSettings from the job config.
    """
    experiment = settings.experiment
    assert experiment is not None
    return experiment


def _require_numeric_lr(settings: TrainingSettings) -> int | float:
    """Extract learning rate from training settings.

    Args:
        settings: Training settings with optimizer configuration.

    Returns:
        Numeric learning rate value.
    """
    optimizer = settings.optimizer.default_optimizer
    assert isinstance(optimizer, AdamWSettings | AdamSettings | LBFGSSettings | MuonSettings)
    lr = optimizer.lr
    assert isinstance(lr, int | float)
    return lr


def test_apply_training_overrides_epochs_batchsize_lr() -> None:
    base = _base_settings()

    new = apply_runtime_overrides(
        base,
        epochs=5,
        batch_size=128,
        learning_rate=0.005,
    )
    training = _require_training(new)
    data = _require_data(new)

    # epochs propagates to training.trainer.max_epochs
    assert training.trainer.max_epochs == 5

    # batch size to data.batch_size
    assert data.batch_size == 128

    # learning rate to training.optimizer.lr
    assert float(_require_numeric_lr(training)) == pytest.approx(0.005)


def test_apply_mlflow_overrides_names() -> None:
    base = _base_settings()

    new = apply_runtime_overrides(base, experiment_name="expA", run_name="run1")
    experiment = _require_experiment(new)
    assert experiment.name == "expA"
    assert experiment.run_name == "run1"


@pytest.mark.parametrize(
    "override_kwargs",
    [
        {"epochs": 0},
        {"epochs": -5},
        {"batch_size": 0},
        {"batch_size": -1},
        {"learning_rate": 0.0},
        {"learning_rate": -0.01},
    ],
)
def test_apply_overrides_pydantic_rejects_invalid_numeric_inputs(
    override_kwargs: dict,
) -> None:
    """Pydantic settings validators must reject out-of-range numeric overrides via patch()."""
    base = _base_settings()

    with pytest.raises(ValidationError):
        apply_runtime_overrides(base, **override_kwargs)


def test_validate_overrides_checks_checkpoint_existence(tmp_path: Path) -> None:
    # checkpoint must exist — only filesystem check, Pydantic validates the rest
    errors = validate_runtime_overrides(checkpoint_path=tmp_path / "missing.ckpt")
    assert any("does not exist" in e for e in errors)

    # existing checkpoint passes
    ckpt = tmp_path / "model.ckpt"
    ckpt.touch()
    errors = validate_runtime_overrides(checkpoint_path=ckpt)
    assert errors == []


def test_training_overrides_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="epcohs"):
        TrainingOverrides.model_validate({"epcohs": 5})
