from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from lightning.pytorch import Trainer

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
    stopping: dict[str, Any] | None = None,
    run: dict[str, Any] | None = None,
    trainer_extra: dict[str, Any] | None = None,
) -> TrainingJobConfig:
    """Build a minimal TrainingJobConfig focused on trainer configuration.

    Args:
        tmp_path: Optional path used as default_root_dir for the trainer.
        default_root_dir: Explicit root dir override (takes precedence over tmp_path).
        logger_name: Logger name to configure, or None for no logger.
        callbacks: Callback dicts to include, or None for none.
        enable_checkpointing: Whether Lightning built-in checkpointing is enabled.
        stopping: Optional TRAINING.stopping overrides (monitor/direction/enabled/patience).
        run: Optional RUN section overrides (merged over the {"type": "train"} default).
        trainer_extra: Optional extra TRAINING.trainer fields (e.g. devices/num_nodes/strategy).

    Returns:
        TrainingJobConfig ready for ``build_trainer``.
    """
    root = default_root_dir or (str(tmp_path) if tmp_path is not None else None)
    trainer_cfg: dict[str, Any] = {"accelerator": "cpu", **(trainer_extra or {})}
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

    training_cfg: dict[str, Any] = {"trainer": trainer_cfg}
    if stopping is not None:
        training_cfg["stopping"] = stopping

    return TrainingJobConfig.model_validate(
        {
            "run": {"type": "train", **(run or {})},
            "model": {"class": "Dummy"},
            "data": {"batch_size": 8, "num_workers": 0},
            "training": training_cfg,
        }
    )


def _callbacks_named(trainer: Trainer | None, name: str) -> list[Any]:
    callbacks = cast("list[object]", cast("Any", trainer).callbacks)
    return [cb for cb in callbacks if type(cb).__name__ == name]


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


# ---------------------------------------------------------------------------
# build_trainer threads settings.run through as the session, so RUN.compute
# (and RUN.precision) actually take effect in the real pipeline.
# ---------------------------------------------------------------------------


def test_build_trainer_applies_explicit_trainer_devices_and_num_nodes(tmp_path: Path) -> None:
    """devices/num_nodes are plain TrainerSettings fields (mirroring Trainer's
    own constructor) — build_trainer must forward them unchanged."""
    settings = _make_trainer_job(
        default_root_dir=str(tmp_path),
        callbacks=[],
        trainer_extra={"devices": 1, "num_nodes": 1},
    )

    trainer = build_trainer(settings)

    assert trainer is not None
    assert trainer.num_nodes == 1


def test_build_trainer_applies_run_compute_environment_selection(tmp_path: Path) -> None:
    """run.compute selects which environment to resolve against; build_trainer
    threads settings.run through as TrainerSettings.build()'s session so this
    (and run.precision) actually take effect in the real pipeline."""
    settings = _make_trainer_job(
        default_root_dir=str(tmp_path),
        callbacks=[],
        run={"compute": {"environment": "local"}},
    )

    trainer = build_trainer(settings)

    assert trainer is not None
    assert trainer.num_nodes == 1


# ---------------------------------------------------------------------------
# Item 2: smart ModelCheckpoint defaults
# ---------------------------------------------------------------------------


def test_build_trainer_auto_injects_default_checkpoint_when_enabled_and_unconfigured(
    tmp_path: Path,
) -> None:
    settings = _make_trainer_job(
        default_root_dir=str(tmp_path),
        callbacks=[],
        enable_checkpointing=True,
    )

    trainer = build_trainer(settings)

    checkpoint_callbacks = _callbacks_named(trainer, "ModelCheckpoint")
    assert len(checkpoint_callbacks) == 1
    cb = checkpoint_callbacks[0]
    assert cb.save_top_k == 1
    # save_last is not set by the default; None is Lightning's own default for
    # ModelCheckpoint(save_last=...) when the argument is omitted.
    assert cb.save_last is None
    assert cb.monitor == "val/loss"
    assert cb.mode == "min"
    assert Path(cb.dirpath).resolve() == (tmp_path / "checkpoints").resolve()


def test_build_trainer_user_supplied_checkpoint_callback_can_opt_into_save_last(
    tmp_path: Path,
) -> None:
    settings = _make_trainer_job(
        default_root_dir=str(tmp_path),
        callbacks=[{"name": "ModelCheckpoint", "save_last": True}],
        enable_checkpointing=True,
    )

    trainer = build_trainer(settings)

    checkpoint_callbacks = _callbacks_named(trainer, "ModelCheckpoint")
    assert len(checkpoint_callbacks) == 1
    cb = checkpoint_callbacks[0]
    assert cb.save_last is True


def test_build_trainer_respects_user_supplied_checkpoint_callback(tmp_path: Path) -> None:
    settings = _make_trainer_job(
        default_root_dir=str(tmp_path),
        callbacks=[{"name": "ModelCheckpoint", "monitor": "val/acc", "mode": "max"}],
        enable_checkpointing=True,
    )

    trainer = build_trainer(settings)

    checkpoint_callbacks = _callbacks_named(trainer, "ModelCheckpoint")
    assert len(checkpoint_callbacks) == 1
    cb = checkpoint_callbacks[0]
    assert cb.monitor == "val/acc"
    assert cb.mode == "max"


def test_build_trainer_auto_injected_checkpoint_reuses_custom_stopping_monitor(
    tmp_path: Path,
) -> None:
    settings = _make_trainer_job(
        default_root_dir=str(tmp_path),
        callbacks=[],
        enable_checkpointing=True,
        stopping={"monitor": "val/acc", "direction": "max"},
    )

    trainer = build_trainer(settings)

    cb = _callbacks_named(trainer, "ModelCheckpoint")[0]
    assert cb.monitor == "val/acc"
    assert cb.mode == "max"


def test_build_trainer_does_not_inject_checkpoint_when_disabled(tmp_path: Path) -> None:
    settings = _make_trainer_job(
        default_root_dir=str(tmp_path),
        callbacks=[],
        enable_checkpointing=False,
    )

    trainer = build_trainer(settings)

    assert _callbacks_named(trainer, "ModelCheckpoint") == []


# ---------------------------------------------------------------------------
# Item 3: opt-in EarlyStopping wiring
# ---------------------------------------------------------------------------


def test_build_trainer_injects_early_stopping_when_stopping_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = _make_trainer_job(
        default_root_dir=None,
        logger_name=None,
        callbacks=[],
        enable_checkpointing=False,
        stopping={"enabled": True},
    )

    trainer = build_trainer(settings)

    assert trainer is not None
    early_stopping_callbacks = _callbacks_named(trainer, "EarlyStopping")
    assert len(early_stopping_callbacks) == 1
    cb = early_stopping_callbacks[0]
    assert cb.monitor == "val/loss"
    assert cb.patience == 10
    assert cb.mode == "min"


def test_build_trainer_does_not_inject_early_stopping_by_default(
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

    assert _callbacks_named(trainer, "EarlyStopping") == []


def test_build_trainer_respects_user_supplied_early_stopping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = _make_trainer_job(
        default_root_dir=None,
        logger_name=None,
        callbacks=[{"name": "EarlyStopping", "monitor": "val/loss", "patience": 3}],
        enable_checkpointing=False,
        stopping={"enabled": True},
    )

    trainer = build_trainer(settings)

    early_stopping_callbacks = _callbacks_named(trainer, "EarlyStopping")
    assert len(early_stopping_callbacks) == 1
    assert early_stopping_callbacks[0].patience == 3


def test_build_trainer_injects_both_checkpoint_and_early_stopping(tmp_path: Path) -> None:
    settings = _make_trainer_job(
        default_root_dir=str(tmp_path),
        callbacks=[],
        enable_checkpointing=True,
        stopping={"enabled": True},
    )

    trainer = build_trainer(settings)

    assert len(_callbacks_named(trainer, "ModelCheckpoint")) == 1
    assert len(_callbacks_named(trainer, "EarlyStopping")) == 1
