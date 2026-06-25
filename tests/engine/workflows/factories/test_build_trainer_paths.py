from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from dlkit.common.errors import WorkflowError
from dlkit.engine.workflows.factories.build_strategy import build_trainer
from dlkit.infrastructure.config.job_config import TrainingJobConfig


def _make_trainer_job(
    tmp_path: Path | None = None,
    *,
    default_root_dir: str | None = None,
    logger_name: str | None = "CSVLogger",
    callbacks: list[dict[str, Any]] | None = None,
    enable_checkpointing: bool | None = None,
) -> TrainingJobConfig:
    """Build a minimal TrainingJobConfig focused on trainer configuration.

    Args:
        tmp_path: Optional path used as default_root_dir for the trainer.
        default_root_dir: Explicit root dir override (takes precedence over tmp_path).
        logger_name: Logger name to configure, or None for no logger.
        callbacks: Callback dicts to include, or None for none.
        enable_checkpointing: Whether Lightning built-in checkpointing is enabled.

    Returns:
        TrainingJobConfig ready for ``build_trainer``.
    """
    root = default_root_dir or (str(tmp_path) if tmp_path is not None else None)
    trainer_cfg: dict[str, Any] = {"accelerator": "cpu"}
    if root is not None:
        trainer_cfg["default_root_dir"] = root
    if logger_name is not None:
        trainer_cfg["logger"] = {"name": logger_name}
    else:
        trainer_cfg["logger"] = {"name": None}
    if callbacks is not None:
        trainer_cfg["callbacks"] = callbacks
    if enable_checkpointing is not None:
        trainer_cfg["enable_checkpointing"] = enable_checkpointing

    return TrainingJobConfig.model_validate(
        {
            "run": {"type": "train"},
            "model": {"class": "Dummy"},
            "data": {"batch_size": 8, "num_workers": 0},
            "training": {"trainer": trainer_cfg},
        }
    )


def test_build_trainer_pins_local_lightning_outputs_without_mlflow(tmp_path: Path) -> None:
    root_dir = tmp_path / "lightning-root"
    root_dir.mkdir()
    settings = _make_trainer_job(
        default_root_dir=str(root_dir),
        logger_name="CSVLogger",
        callbacks=[{"name": "ModelCheckpoint"}],
    )

    trainer = build_trainer(settings)

    assert trainer is not None
    configured_root = settings.training.trainer.default_root_dir
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
    settings = _make_trainer_job(
        default_root_dir=None,
        logger_name="CSVLogger",
        callbacks=[{"name": "ModelCheckpoint"}],
    )

    with pytest.raises(WorkflowError, match="default_root_dir is required"):
        build_trainer(settings)


def test_build_trainer_requires_default_root_when_checkpointing_enabled() -> None:
    settings = _make_trainer_job(
        default_root_dir=None,
        logger_name=None,
        callbacks=[],
        enable_checkpointing=True,
    )

    with pytest.raises(WorkflowError, match="default_root_dir is required"):
        build_trainer(settings)


def test_build_trainer_allows_noop_mode_without_default_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = _make_trainer_job(
        default_root_dir=None,
        logger_name=None,
        callbacks=[],
        enable_checkpointing=False,
    )

    trainer = build_trainer(settings)

    assert trainer is not None
    callbacks = cast("list[object]", cast("Any", trainer).callbacks)
    checkpoint_callbacks = [cb for cb in callbacks if type(cb).__name__ == "ModelCheckpoint"]
    assert not checkpoint_callbacks
    assert not (tmp_path / "checkpoints").exists()
