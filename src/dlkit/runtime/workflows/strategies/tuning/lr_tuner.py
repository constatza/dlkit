"""Learning rate tuning service using Lightning's Tuner."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

from lightning.pytorch.tuner import Tuner

from dlkit.tools.utils.logging_config import get_logger

logger = get_logger(__name__)

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
        logger.debug(
            "LR tuner config: min_lr={} max_lr={} num_training={} mode={}",
            settings.min_lr,
            settings.max_lr,
            settings.num_training,
            settings.mode,
        )

        # Import torch.serialization for safe globals registration
        import torch.serialization

        # Collect all dlkit config classes that might be in checkpoints
        safe_classes = self._get_safe_globals()

        tuner = Tuner(trainer)

        # Run learning rate finder with safe globals context for PyTorch 2.6+
        # This allows LR finder to save/restore checkpoints with dlkit config classes
        with torch.serialization.safe_globals(safe_classes):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*weights_only.*", category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*",
                    category=UserWarning,
                )
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

    def _get_safe_globals(self) -> list[type]:
        """Get list of dlkit classes to register as safe globals for checkpoint loading.

        Returns:
            list[type]: List of classes that should be allowed in checkpoint loading
        """
        from dlkit.tools.config.components.model_components import (
            WrapperComponentSettings,
            ModelComponentSettings,
            MetricComponentSettings,
            LossComponentSettings,
        )
        from dlkit.tools.config.optimizer_settings import OptimizerSettings, SchedulerSettings
        from dlkit.tools.config.datamodule_settings import DataModuleSettings
        from dlkit.tools.config.dataset_settings import DatasetSettings, IndexSplitSettings
        from dlkit.tools.config.dataloader_settings import DataloaderSettings
        from dlkit.tools.config.lr_tuner_settings import LRTunerSettings
        from dlkit.tools.config.mlflow_settings import MLflowSettings
        from dlkit.tools.config.core.base_settings import BasicSettings
        from dlkit.tools.config.general_settings import GeneralSettings
        from dlkit.tools.config.training_settings import TrainingSettings
        from dlkit.tools.config.session_settings import SessionSettings
        from dlkit.tools.config.paths_settings import PathsSettings

        return [
            # Base settings
            BasicSettings,
            GeneralSettings,
            TrainingSettings,
            SessionSettings,
            PathsSettings,
            # Model component settings
            WrapperComponentSettings,
            ModelComponentSettings,
            MetricComponentSettings,
            LossComponentSettings,
            # Training settings
            OptimizerSettings,
            SchedulerSettings,
            # Data settings
            DataModuleSettings,
            DatasetSettings,
            IndexSplitSettings,
            DataloaderSettings,
            # Other settings
            LRTunerSettings,
            MLflowSettings,
        ]
