"""Learning rate management concern for ProcessingLightningWrapper.

Encapsulates learning rate resolution and synchronization logic.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ILearningRateManager(Protocol):
    """Protocol for learning rate management.

    Implementations resolve LR from config, trainer, or both.
    """

    def get_lr(self) -> float | None:
        """Get current learning rate.

        Returns:
            Learning rate as float, or None if unavailable.
        """
        ...

    def set_lr(self, value: float) -> None:
        """Set learning rate in config and trainer (if attached).

        Args:
            value: New learning rate value.
        """
        ...

    def sync_hparam(self) -> None:
        """Synchronize hparams with current learning rate.

        Updates self.hparams['lr'] and self.hparams['learning_rate'].
        """
        ...


class ConfigLearningRateManager:
    """Manages learning rate from optimizer settings and attached trainer.

    Resolves LR via: config optimizer.lr → trainer.optimizers[0] → None.
    Allows runtime override and synchronization with hparams.
    """

    def __init__(self, module: Any) -> None:
        """Initialize with a reference to the LightningModule.

        Args:
            module: The LightningModule instance.
        """
        self._module = module

    def get_lr(self) -> float | None:
        """Get current learning rate.

        First checks optimizer settings, then trainer optimizers.

        Returns:
            Learning rate as float, or None if unavailable.
        """
        optimizer_settings = getattr(self._module, "optimizer", None)
        raw_lr = getattr(optimizer_settings, "lr", None)
        if isinstance(raw_lr, (float, int)):
            return float(raw_lr)

        trainer = self._get_attached_trainer()
        if trainer and getattr(trainer, "optimizers", None):
            try:
                return float(trainer.optimizers[0].param_groups[0]["lr"])
            except KeyError, IndexError, TypeError, ValueError:
                return None
        return None

    def set_lr(self, value: float) -> None:
        """Set learning rate in config and trainer.

        Args:
            value: New learning rate value.
        """
        from dlkit.infrastructure.config.core.updater import update_settings

        numeric = float(value)
        self._module.optimizer = update_settings(self._module.optimizer, {"lr": numeric})

        trainer = self._get_attached_trainer()
        if trainer and getattr(trainer, "optimizers", None):
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = numeric

        self.sync_hparam()

    def sync_hparam(self) -> None:
        """Synchronize stored hyperparameters with the current learning rate.

        Updates hparams['lr'] and hparams['learning_rate'].
        """
        lr_value = self.get_lr()
        if not hasattr(self._module, "hparams"):
            return
        self._module.hparams["lr"] = lr_value
        self._module.hparams["learning_rate"] = lr_value

    def _get_attached_trainer(self) -> Any:
        """Return attached trainer without triggering Lightning errors.

        Checks both _trainer and _fabric attributes.

        Returns:
            Trainer instance or shim, or None if unavailable.
        """
        trainer = getattr(self._module, "_trainer", None)
        if trainer is not None:
            return trainer

        fabric = getattr(self._module, "_fabric", None)
        if fabric is not None:
            try:
                from lightning.pytorch.core.module import _TrainerFabricShim

                return _TrainerFabricShim(fabric=fabric)
            except Exception:
                return None
        return None
