from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from dlkit.engine.workflows.factories.build_strategy import build_trainer
from dlkit.infrastructure.config.environment import env
from dlkit.infrastructure.config.session_settings import SessionSettings
from dlkit.infrastructure.config.trainer_settings import (
    CallbackSettings,
    LoggerSettings,
    TrainerSettings,
)
from dlkit.infrastructure.config.training_settings import TrainingSettings
from dlkit.infrastructure.config.workflow_configs import TrainingWorkflowConfig


def test_build_trainer_pins_local_lightning_outputs_without_mlflow() -> None:
    settings = TrainingWorkflowConfig(
        SESSION=SessionSettings(workflow="train"),
        TRAINING=TrainingSettings(
            trainer=TrainerSettings(
                accelerator="cpu",
                logger=LoggerSettings(name="CSVLogger"),
                callbacks=(CallbackSettings(name="ModelCheckpoint"),),
            )
        ),
    )

    trainer = build_trainer(settings)

    assert trainer is not None
    expected_root = (env.get_internal_dir_path() / "lightning").resolve()
    assert Path(trainer.default_root_dir).resolve() == expected_root

    csv_logger = trainer.loggers[0]
    assert csv_logger.save_dir is not None
    assert Path(csv_logger.save_dir).resolve() == expected_root

    callbacks = cast("list[object]", cast("Any", trainer).callbacks)
    checkpoint_callbacks = [cb for cb in callbacks if type(cb).__name__ == "ModelCheckpoint"]
    assert checkpoint_callbacks
    checkpoint_dir = checkpoint_callbacks[0].dirpath
    assert checkpoint_dir is not None
    assert Path(checkpoint_dir).resolve() == expected_root / "checkpoints"
