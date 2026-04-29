from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dlkit.engine.workflows.entrypoints._overrides import (
    apply_runtime_overrides,
    validate_runtime_overrides,
)
from dlkit.infrastructure.config.dataloader_settings import DataloaderSettings
from dlkit.infrastructure.config.datamodule_settings import DataModuleSettings
from dlkit.infrastructure.config.general_settings import GeneralSettings
from dlkit.infrastructure.config.mlflow_settings import MLflowSettings
from dlkit.infrastructure.config.session_settings import SessionSettings
from dlkit.infrastructure.config.training_settings import TrainingSettings


def _base_settings() -> GeneralSettings:
    return GeneralSettings(
        SESSION=SessionSettings(workflow="train"),
        TRAINING=TrainingSettings(),
        DATAMODULE=DataModuleSettings(dataloader=DataloaderSettings()),
        MLFLOW=MLflowSettings(),
    )


def _require_training(settings: GeneralSettings) -> TrainingSettings:
    training = settings.TRAINING
    assert training is not None
    return training


def _require_datamodule(settings: GeneralSettings) -> DataModuleSettings:
    datamodule = settings.DATAMODULE
    assert datamodule is not None
    return datamodule


def _require_mlflow(settings: GeneralSettings) -> MLflowSettings:
    mlflow = settings.MLFLOW
    assert mlflow is not None
    return mlflow


def _require_numeric_lr(settings: TrainingSettings) -> int | float:
    lr = settings.optimizer.lr
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
    datamodule = _require_datamodule(new)

    # epochs propagates to TRAINING.epochs and TRAINING.trainer.max_epochs
    assert training.epochs == 5
    assert training.trainer.max_epochs == 5

    # batch size to DATAMODULE.dataloader.batch_size
    assert datamodule.dataloader.batch_size == 128

    # learning rate to TRAINING.optimizer.lr
    assert float(_require_numeric_lr(training)) == pytest.approx(0.005)


def test_apply_mlflow_overrides_names() -> None:
    base = _base_settings()

    new = apply_runtime_overrides(base, experiment_name="expA", run_name="run1")
    mlflow = _require_mlflow(new)
    assert mlflow.experiment_name == "expA"
    assert mlflow.run_name == "run1"


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
