"""Tests for checkpoint-callback lookup helpers."""

from __future__ import annotations

from unittest.mock import Mock

from lightning.pytorch.callbacks import ModelCheckpoint

from dlkit.engine.training.checkpoint_utils import (
    find_checkpoint_callback,
    find_checkpoint_callbacks,
)


def _mock_trainer(callbacks: list[object], checkpoint_callback: object | None = None) -> Mock:
    trainer = Mock()
    trainer.callbacks = callbacks
    trainer.checkpoint_callback = checkpoint_callback
    return trainer


class TestFindCheckpointCallback:
    """Tests for find_checkpoint_callback / find_checkpoint_callbacks."""

    def test_prefers_trainer_checkpoint_callback(self) -> None:
        primary = ModelCheckpoint()
        trainer = _mock_trainer(callbacks=[primary], checkpoint_callback=primary)

        assert find_checkpoint_callback(trainer) is primary

    def test_falls_back_to_scanning_callbacks_when_checkpoint_callback_unset(self) -> None:
        cb = ModelCheckpoint()
        trainer = _mock_trainer(callbacks=[cb], checkpoint_callback=None)

        assert find_checkpoint_callback(trainer) is cb

    def test_returns_none_for_mock_checkpoint_callback(self) -> None:
        """A Mock() trainer auto-vivifies `.checkpoint_callback` as a truthy
        non-ModelCheckpoint object; isinstance() must reject it, not a truthy check."""
        trainer = Mock()
        trainer.callbacks = []

        assert find_checkpoint_callback(trainer) is None

    def test_returns_none_when_no_checkpoint_callback_configured(self) -> None:
        trainer = _mock_trainer(callbacks=[], checkpoint_callback=None)

        assert find_checkpoint_callback(trainer) is None

    def test_find_all_returns_every_match(self) -> None:
        cb1 = ModelCheckpoint()
        cb2 = ModelCheckpoint()
        trainer = _mock_trainer(callbacks=[cb1, object(), cb2], checkpoint_callback=None)

        assert find_checkpoint_callbacks(trainer) == [cb1, cb2]

    def test_find_all_returns_empty_list_when_none_configured(self) -> None:
        trainer = _mock_trainer(callbacks=[object()], checkpoint_callback=None)

        assert find_checkpoint_callbacks(trainer) == []
