"""Training settings - flattened top-level configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, PositiveInt

from .core.base_settings import BasicSettings
from .lr_tuner_settings import LRTunerSettings
from .model_components import LossComponentSettings, MetricComponentSettings
from .optimizer_settings import OptimizerSettings, SchedulerSettings
from .trainer_settings import TrainerSettings


class TrainingSettings(BasicSettings):
    """Top-level Training configuration with library-specific nested settings.

    Flattened from mode architecture to top-level for easier access.
    Replaces: settings.SESSION.training (minus plugins and data_pipeline)
    New usage: settings.TRAINING

    Contains core training settings with nested library-specific configurations.
    Plugins (MLflow, Optuna) are moved to top-level settings.
    Data configuration is moved to top-level DATAMODULE and DATASET.

    Args:
        trainer: PyTorch Lightning trainer settings (library-specific - kept nested)
        optimizer: Optimizer and scheduler settings (library-specific - kept nested)
        scheduler: Learning rate scheduler settings
        resume_from_checkpoint: Checkpoint to resume training from (full state)
        epochs: Number of training epochs
        patience: Early stopping patience
        monitor_metric: Metric to monitor for early stopping
        mode: Monitoring mode (min/max)
        loss_function: Loss function configuration
        metrics: Metrics to compute during training
    """

    # Library-specific configurations (kept nested as requested)
    trainer: TrainerSettings = Field(
        default_factory=TrainerSettings, description="PyTorch Lightning trainer settings"
    )
    optimizer: OptimizerSettings = Field(
        default_factory=OptimizerSettings, description="Optimizer settings"
    )
    scheduler: SchedulerSettings | None = Field(
        default=None, description="Learning rate scheduler settings"
    )

    # Checkpoint for resuming training
    resume_from_checkpoint: Path | str | None = Field(
        default=None,
        description=(
            "Checkpoint to resume training from. Includes full training state: "
            "model weights, optimizer state, scheduler state, epoch, global step, etc. "
            "This checkpoint is used by PyTorch Lightning's Trainer.fit(ckpt_path=...)"
        ),
    )

    # Core training parameters (flattened)
    epochs: PositiveInt = Field(default=100, description="Number of training epochs")
    patience: int = Field(default=10, description="Early stopping patience")
    monitor_metric: str = Field(
        default="val_loss", description="Metric to monitor for early stopping"
    )
    mode: str = Field(default="min", description="Monitoring mode (min/max for early stopping)")
    lr_tuner: LRTunerSettings | None = Field(
        default=None,
        description="Learning rate tuner settings. If configured, automatic LR tuning runs before training.",
    )

    # Loss function and metrics configuration
    loss_function: LossComponentSettings = Field(
        default_factory=LossComponentSettings,
        description="Loss function settings for training and validation",
    )
    metrics: tuple[MetricComponentSettings, ...] = Field(
        default=tuple(), description="List of metrics to compute on the model at test time"
    )

    @property
    def has_scheduler(self) -> bool:
        """Check if scheduler is configured.

        Returns:
            bool: True if scheduler settings are provided
        """
        return self.scheduler is not None

    def get_trainer_config(self) -> dict:
        """Get raw Trainer kwargs without signature filtering.

        Uses ``TrainerSettings.to_dict()`` (excludes None/unset/meta fields) and
        adds ``max_epochs`` from the top-level training settings. This avoids
        silently dropping invalid keys; Lightning will raise on invalid kwargs.
        """
        config = self.trainer.to_dict()
        config.update({"max_epochs": self.epochs})
        return config
