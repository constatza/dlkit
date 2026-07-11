"""Shared ModelCheckpoint-callback lookup, used across engine subpackages.

Lives in `engine.training` (not `engine.adapters.lightning`) because the
curated dependency graph already has `engine.adapters -> engine.training`
and `engine.tracking -> engine.training` edges — placing this here lets both
of those subpackages depend on it without creating a cycle back into
`engine.adapters`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import ModelCheckpoint

if TYPE_CHECKING:
    from lightning.pytorch import Trainer


def find_checkpoint_callbacks(trainer: Trainer) -> list[ModelCheckpoint]:
    """Return every ModelCheckpoint callback configured on a trainer.

    Args:
        trainer: PyTorch Lightning trainer.

    Returns:
        All `ModelCheckpoint` instances in `trainer.callbacks`, in order.
    """
    callbacks = getattr(trainer, "callbacks", None) or []
    return [cb for cb in callbacks if isinstance(cb, ModelCheckpoint)]


def find_checkpoint_callback(trainer: Trainer) -> ModelCheckpoint | None:
    """Return the primary ModelCheckpoint callback configured on a trainer.

    Prefers `trainer.checkpoint_callback` (Lightning's own resolved primary
    callback), falling back to the first match among `trainer.callbacks`.

    Args:
        trainer: PyTorch Lightning trainer.

    Returns:
        The primary `ModelCheckpoint` callback, or `None` if none is configured.
    """
    primary = getattr(trainer, "checkpoint_callback", None)
    if isinstance(primary, ModelCheckpoint):
        return primary

    matches = find_checkpoint_callbacks(trainer)
    return matches[0] if matches else None
