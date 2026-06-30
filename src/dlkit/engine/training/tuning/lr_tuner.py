"""Learning rate tuning service using Lightning's Tuner."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch.tuner.tuning import Tuner

from dlkit.engine.training.tuning.transform_fitting import (
    IFittableTransformer,
    IHasBatchTransformer,
    fit_if_needed,
)
from dlkit.infrastructure.utils.logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from lightning.pytorch import LightningDataModule, LightningModule, Trainer

    from dlkit.infrastructure.config.lr_tuner_settings import LRTunerSettings


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

        if datamodule is not None:
            # Trainer.datamodule is a mutable attribute; cast to Any to satisfy ty stubs
            cast(Any, trainer).datamodule = datamodule

            if isinstance(model, IHasBatchTransformer):
                # Lightning's lr_find() strips trainer.callbacks down to its own
                # internal callback before running the scan loop, so dlkit's
                # TransformFittingCallback.on_fit_start never runs here — fit
                # explicitly instead of relying on Lightning's callback lifecycle.
                # nn.Module.__getattr__'s stub widens batch_transformer's static
                # type; cast to satisfy ty.
                fit_if_needed(
                    cast(IFittableTransformer, model.batch_transformer),
                    datamodule.train_dataloader(),
                    device=model.device,
                )

        tuner = Tuner(trainer)
        original_callbacks = self._snapshot_callbacks(trainer)

        # Run learning rate finder with safe globals context for PyTorch 2.6+
        # This allows LR finder to save/restore checkpoints with dlkit config classes
        try:
            with torch.serialization.safe_globals(cast(Any, safe_classes)):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=".*weights_only.*", category=UserWarning
                    )
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
                    suggested_lr = lr_finder.suggestion() if lr_finder is not None else None
        except MisconfigurationException as error:
            raise RuntimeError(
                "Learning rate tuning failed because Lightning's LR finder only supports "
                "a single optimizer. Disable [TRAINING.lr_tuner] or use a single-stage "
                "optimizer policy."
            ) from error
        except IndexError as error:
            raise RuntimeError(
                "Learning rate tuning failed inside Lightning before training started. "
                "This usually means the model's optimizer configuration is incompatible "
                "with Lightning's LR finder. Configure a scheduler or disable "
                "[TRAINING.lr_tuner]."
            ) from error
        finally:
            self._restore_callbacks(trainer, original_callbacks)

        if suggested_lr is None:
            raise RuntimeError(
                "Learning rate tuner failed to suggest a learning rate. "
                "This may happen if the loss curve is too noisy or flat. "
                "Try adjusting tuner settings (min_lr, max_lr, num_training) "
                "or using a manual learning rate instead."
            )

        logger.info("Learning rate tuner suggested: {}", suggested_lr)

        return float(suggested_lr)

    def _snapshot_callbacks(self, trainer: Trainer) -> list[object] | None:
        """Capture the trainer callback list so temporary tuner callbacks can be removed."""
        callbacks = getattr(trainer, "callbacks", None)
        if callbacks is None:
            return None
        return list(cast(list[object], callbacks))

    def _restore_callbacks(
        self,
        trainer: Trainer,
        original_callbacks: list[object] | None,
    ) -> None:
        """Restore the trainer callback list after LR finder runs or fails."""
        if original_callbacks is None:
            return
        # Lightning Trainer.callbacks is always a plain mutable list attribute;
        # cast to Any to satisfy ty (Trainer stub does not declare callbacks).
        cast(Any, trainer).callbacks = list(original_callbacks)

    def _get_safe_globals(self) -> list[type[object]]:
        """Get list of dlkit classes to register as safe globals for checkpoint loading.

        Returns:
            list[type]: List of classes that should be allowed in checkpoint loading
        """
        from dlkit.infrastructure.config.core.base_settings import BasicSettings
        from dlkit.infrastructure.config.data_settings import DataModuleSelector, DataSettings
        from dlkit.infrastructure.config.dataloader_settings import DataloaderSettings
        from dlkit.infrastructure.config.job_config import (
            InferenceJobConfig,
            JobConfig,
            SearchJobConfig,
            TrainingJobConfig,
        )
        from dlkit.infrastructure.config.lr_tuner_settings import LRTunerSettings
        from dlkit.infrastructure.config.model_components import (
            LossComponentSettings,
            MetricComponentSettings,
            ModelComponentSettings,
            WrapperComponentSettings,
        )
        from dlkit.infrastructure.config.optimizer_settings import (
            OptimizerSettings,
            SchedulerSettings,
        )
        from dlkit.infrastructure.config.paths_settings import PathsSettings
        from dlkit.infrastructure.config.run_settings import RunSettings
        from dlkit.infrastructure.config.split_settings import IndexSplitSettings
        from dlkit.infrastructure.config.tracking_settings import TrackingSettings
        from dlkit.infrastructure.config.training_settings import StoppingSettings, TrainingSettings

        return [
            # Base settings
            BasicSettings,
            TrainingSettings,
            StoppingSettings,
            PathsSettings,
            RunSettings,
            # JobConfig classes
            JobConfig,
            TrainingJobConfig,
            InferenceJobConfig,
            SearchJobConfig,
            # Model settings
            ModelComponentSettings,
            WrapperComponentSettings,
            MetricComponentSettings,
            LossComponentSettings,
            # Training settings
            OptimizerSettings,
            SchedulerSettings,
            # Data settings
            DataSettings,
            DataModuleSelector,
            IndexSplitSettings,
            DataloaderSettings,
            # Tracking
            TrackingSettings,
            # Other settings
            LRTunerSettings,
        ]
