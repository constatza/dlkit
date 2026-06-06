from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from dlkit.common.errors import WorkflowError
from dlkit.engine.workflows.factories.build_strategy import build_trainer
from dlkit.infrastructure.config.session_settings import SessionSettings
from dlkit.infrastructure.config.trainer_settings import (
    CallbackSettings,
    LoggerSettings,
    TrainerSettings,
)
from dlkit.infrastructure.config.training_settings import TrainingSettings
from dlkit.infrastructure.config.workflow_configs import TrainingWorkflowConfig


def test_build_trainer_pins_local_lightning_outputs_without_mlflow(tmp_path: Path) -> None:
    root_dir = tmp_path / "lightning-root"
    root_dir.mkdir()
    settings = TrainingWorkflowConfig(
        SESSION=SessionSettings(workflow="train"),
        TRAINING=TrainingSettings(
            trainer=TrainerSettings(
                accelerator="cpu",
                default_root_dir=root_dir,
                logger=LoggerSettings(name="CSVLogger"),
                callbacks=(CallbackSettings(name="ModelCheckpoint"),),
            )
        ),
    )

    trainer = build_trainer(settings)

    assert trainer is not None
    configured_root = settings.TRAINING.trainer.default_root_dir
    assert configured_root is not None
    expected_root = Path(configured_root).resolve()
    assert Path(trainer.default_root_dir).resolve() == expected_root

    csv_logger = trainer.loggers[0]
    assert csv_logger.save_dir is not None
    assert Path(csv_logger.save_dir).resolve() == expected_root

    callbacks = cast("list[object]", cast("Any", trainer).callbacks)
    checkpoint_callbacks = [cb for cb in callbacks if type(cb).__name__ == "ModelCheckpoint"]
    assert checkpoint_callbacks
    checkpoint_callback = cast("Any", checkpoint_callbacks[0])
    checkpoint_dir = checkpoint_callback.dirpath
    assert checkpoint_dir is not None
    assert Path(checkpoint_dir).resolve() == expected_root / "checkpoints"


def test_build_trainer_requires_default_root_for_local_output_producers() -> None:
    settings = TrainingWorkflowConfig(
        SESSION=SessionSettings(workflow="train"),
        TRAINING=TrainingSettings(
            trainer=TrainerSettings(
                accelerator="cpu",
                logger=LoggerSettings(name="CSVLogger"),
                callbacks=(CallbackSettings(name="ModelCheckpoint"),),
                default_root_dir=None,
            )
        ),
    )

    with pytest.raises(WorkflowError, match="default_root_dir is required"):
        build_trainer(settings)


def test_build_trainer_requires_default_root_when_checkpointing_enabled() -> None:
    settings = TrainingWorkflowConfig(
        SESSION=SessionSettings(workflow="train"),
        TRAINING=TrainingSettings(
            trainer=TrainerSettings(
                accelerator="cpu",
                enable_checkpointing=True,
                default_root_dir=None,
            )
        ),
    )

    with pytest.raises(WorkflowError, match="default_root_dir is required"):
        build_trainer(settings)


def test_build_trainer_allows_noop_mode_without_default_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = TrainingWorkflowConfig(
        SESSION=SessionSettings(workflow="train"),
        TRAINING=TrainingSettings(
            trainer=TrainerSettings(
                accelerator="cpu",
                enable_checkpointing=False,
                logger=LoggerSettings(name=None),
                callbacks=(),
                default_root_dir=None,
            )
        ),
    )

    trainer = build_trainer(settings)

    assert trainer is not None
    callbacks = cast("list[object]", cast("Any", trainer).callbacks)
    checkpoint_callbacks = [cb for cb in callbacks if type(cb).__name__ == "ModelCheckpoint"]
    assert not checkpoint_callbacks
    assert not (tmp_path / "checkpoints").exists()
