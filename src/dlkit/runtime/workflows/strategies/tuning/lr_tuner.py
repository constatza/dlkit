"""Learning rate tuning service using Lightning's Tuner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lightning.pytorch.tuner import Tuner
from loguru import logger

if TYPE_CHECKING:
    from lightning.pytorch import LightningDataModule, LightningModule, Trainer
    from dlkit.tools.config.lr_tuner_settings import LRTunerSettings


class LRTuner:
    """Service for automatic learning rate tuning using Lightning's Tuner.

    This class wraps PyTorch Lightning's Tuner to find an optimal learning rate
    before training begins. It uses the lr_find algorithm which performs a learning
    rate range test by training the model with exponentially or linearly increasing
    learning rates.

    The tuner suggests an optimal learning rate based on the loss curve, typically
    selecting the learning rate with the steepest negative gradient.
    """

    def tune(
        self,
        trainer: Trainer,
        model: LightningModule,
        settings: LRTunerSettings,
        datamodule: LightningDataModule | None = None,
    ) -> float:
        """Run learning rate finder and return suggested learning rate.

        Args:
            trainer: PyTorch Lightning trainer instance
            model: Lightning module to tune
            settings: LR tuner configuration settings
            datamodule: Optional datamodule for training data

        Returns:
            float: Suggested optimal learning rate

        Raises:
            RuntimeError: If tuner fails to suggest a learning rate
        """
        logger.info("Starting automatic learning rate tuning...")
        logger.info(
            f"LR tuner config: min_lr={settings.min_lr}, max_lr={settings.max_lr}, "
            f"num_training={settings.num_training}, mode={settings.mode}"
        )

        tuner = Tuner(trainer)

        # Run learning rate finder with configured parameters
        lr_finder = tuner.lr_find(
            model,
            datamodule=datamodule,
            min_lr=settings.min_lr,
            max_lr=settings.max_lr,
            num_training=settings.num_training,
            mode=settings.mode,
            early_stop_threshold=settings.early_stop_threshold,
        )

        # Get suggested learning rate
        suggested_lr = lr_finder.suggestion()

        if suggested_lr is None:
            raise RuntimeError(
                "Learning rate tuner failed to suggest a learning rate. "
                "This may happen if the loss curve is too noisy or flat. "
                "Try adjusting tuner settings (min_lr, max_lr, num_training) "
                "or using a manual learning rate instead."
            )

        logger.info(f"Learning rate tuner suggested: {suggested_lr}")

        return suggested_lr
