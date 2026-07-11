"""Tests for CheckpointDirRouter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from lightning.pytorch.callbacks import ModelCheckpoint

from dlkit.engine.adapters.lightning.callbacks import CheckpointDirRouter


def _mock_trainer(callbacks: list[object], checkpoint_callback: object | None = None) -> Mock:
    trainer = Mock()
    trainer.callbacks = callbacks
    trainer.checkpoint_callback = checkpoint_callback
    return trainer


class TestCheckpointDirRouter:
    """Tests for CheckpointDirRouter.on_fit_start redirecting unset dirpaths."""

    def test_redirects_all_unset_model_checkpoints(self, tmp_path: Path) -> None:
        cb1 = ModelCheckpoint()
        cb2 = ModelCheckpoint()
        trainer = _mock_trainer(callbacks=[cb1, cb2], checkpoint_callback=None)
        checkpoint_dir = tmp_path / "checkpoints"
        router = CheckpointDirRouter(checkpoint_dir=checkpoint_dir)

        router.on_fit_start(trainer, Mock())

        assert cb1.dirpath == str(checkpoint_dir)
        assert cb2.dirpath == str(checkpoint_dir)
        assert checkpoint_dir.exists()

    def test_does_not_override_already_set_dirpath(self, tmp_path: Path) -> None:
        existing_dir = str(tmp_path / "existing")
        cb = ModelCheckpoint(dirpath=existing_dir)
        trainer = _mock_trainer(callbacks=[cb], checkpoint_callback=None)
        router = CheckpointDirRouter(checkpoint_dir=tmp_path / "checkpoints")

        router.on_fit_start(trainer, Mock())

        assert cb.dirpath == existing_dir

    def test_noop_when_checkpoint_dir_is_none(self, tmp_path: Path) -> None:
        cb = ModelCheckpoint()
        trainer = _mock_trainer(callbacks=[cb], checkpoint_callback=None)
        router = CheckpointDirRouter(checkpoint_dir=None)

        router.on_fit_start(trainer, Mock())

        assert cb.dirpath is None
