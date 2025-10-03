"""Training settings - flattened top-level configuration."""

from __future__ import annotations

from pydantic import Field

from .core.base_settings import BasicSettings
from .trainer_settings import TrainerSettings
from .optimizer_settings import OptimizerSettings, SchedulerSettings
from .components.model_components import LossComponentSettings, MetricComponentSettings


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
        epochs: Number of training epochs
        patience: Early stopping patience
        monitor_metric: Metric to monitor for early stopping
        mode: Monitoring mode (min/max)
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

    # Core training parameters (flattened)
    epochs: int = Field(default=100, description="Number of training epochs")
    patience: int = Field(default=10, description="Early stopping patience")
    monitor_metric: str = Field(
        default="val_loss", description="Metric to monitor for early stopping"
    )
    mode: str = Field(default="min", description="Monitoring mode (min/max for early stopping)")

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

    @property
    def max_epochs(self) -> int:
        """Get maximum number of epochs (alias for epochs).

        Returns:
            int: Maximum training epochs
        """
        return self.epochs

    def get_trainer_config(self) -> dict:
        """Get raw Trainer kwargs without signature filtering.

        Uses ``TrainerSettings.to_dict()`` (excludes None/unset/meta fields) and
        adds ``max_epochs`` from the top-level training settings. This avoids
        silently dropping invalid keys; Lightning will raise on invalid kwargs.
        """
        config = self.trainer.to_dict()
        config.update({"max_epochs": self.epochs})
        return config
