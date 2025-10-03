from __future__ import annotations

from pathlib import Path

import pytest

from dlkit.interfaces.api.overrides.manager import BasicOverrideManager
from dlkit.tools.config.general_settings import GeneralSettings
from dlkit.tools.config.training_settings import TrainingSettings
from dlkit.tools.config.datamodule_settings import DataModuleSettings
from dlkit.tools.config.dataloader_settings import DataloaderSettings
from dlkit.tools.config.mlflow_settings import MLflowSettings
from dlkit.tools.config.session_settings import SessionSettings


def _base_settings() -> GeneralSettings:
    return GeneralSettings(
        SESSION=SessionSettings(inference=False),
        TRAINING=TrainingSettings(),
        DATAMODULE=DataModuleSettings(dataloader=DataloaderSettings()),
        MLFLOW=MLflowSettings(),
    )


def test_apply_training_overrides_epochs_batchsize_lr(tmp_path: Path) -> None:
    mgr = BasicOverrideManager()
    base = _base_settings()

    new = mgr.apply_overrides(
        base,
        epochs=5,
        batch_size=128,
        learning_rate=0.005,
    )

    # epochs propagates to TRAINING.epochs and TRAINING.trainer.max_epochs
    assert new.TRAINING.epochs == 5
    assert new.TRAINING.trainer.max_epochs == 5

    # batch size to DATAMODULE.batch_size and nested dataloader
    assert new.DATAMODULE.batch_size == 128
    assert new.DATAMODULE.dataloader.batch_size == 128

    # learning rate to TRAINING.optimizer.lr
    assert float(new.TRAINING.optimizer.lr) == pytest.approx(0.005)


def test_apply_mlflow_overrides_names() -> None:
    mgr = BasicOverrideManager()
    base = _base_settings()

    new = mgr.apply_overrides(base, experiment_name="expA", run_name="run1")
    assert new.MLFLOW.client.experiment_name == "expA"
    assert new.MLFLOW.client.run_name == "run1"


def test_validate_overrides_checks_values_and_plugins(tmp_path: Path) -> None:
    mgr = BasicOverrideManager()
    base = _base_settings()

    # checkpoint must exist
    errors = mgr.validate_overrides(base, checkpoint_path=tmp_path / "missing.ckpt")
    assert any("does not exist" in e for e in errors)

    # numeric must be positive
    errors = mgr.validate_overrides(
        base, epochs=0, batch_size=-1, learning_rate=0, mlflow_port=-10, trials=0
    )
    assert len(errors) >= 5
