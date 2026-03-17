from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

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


def test_apply_training_overrides_epochs_batchsize_lr() -> None:
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

    # batch size to DATAMODULE.dataloader.batch_size
    assert new.DATAMODULE.dataloader.batch_size == 128

    # learning rate to TRAINING.optimizer.lr
    assert float(new.TRAINING.optimizer.lr) == pytest.approx(0.005)


def test_apply_mlflow_overrides_names() -> None:
    mgr = BasicOverrideManager()
    base = _base_settings()

    new = mgr.apply_overrides(base, experiment_name="expA", run_name="run1")
    assert new.MLFLOW.experiment_name == "expA"
    assert new.MLFLOW.run_name == "run1"


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
    mgr = BasicOverrideManager()
    base = _base_settings()

    with pytest.raises(ValidationError):
        mgr.apply_overrides(base, **override_kwargs)


def test_validate_overrides_checks_checkpoint_existence(tmp_path: Path) -> None:
    mgr = BasicOverrideManager()
    base = _base_settings()

    # checkpoint must exist — only filesystem check, Pydantic validates the rest
    errors = mgr.validate_overrides(base, checkpoint_path=tmp_path / "missing.ckpt")
    assert any("does not exist" in e for e in errors)

    # existing checkpoint passes
    ckpt = tmp_path / "model.ckpt"
    ckpt.touch()
    errors = mgr.validate_overrides(base, checkpoint_path=ckpt)
    assert errors == []
