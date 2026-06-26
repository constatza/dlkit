"""Training settings - flattened top-level configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from .core.base_settings import BasicSettings
from .lr_tuner_settings import LRTunerSettings
from .model_components import LossComponentSettings, MetricComponentSettings
from .optimizer_policy import OptimizerPolicySettings
from .trainer_settings import TrainerSettings


class StoppingSettings(BasicSettings):
    """Early stopping configuration.

    Args:
        monitor: Metric name to monitor for early stopping.
        patience: Number of epochs with no improvement before stopping.
        direction: Whether lower ("min") or higher ("max") metric values are better.
    """

    monitor: str = "val/loss"
    patience: int = 10
    direction: Literal["min", "max"] = "min"


class TrainingSettings(BasicSettings):
    """Top-level Training configuration with library-specific nested settings.

    Flattened from mode architecture to top-level for easier access.
    Replaces the old session-scoped training subtree (minus plugins and data_pipeline).

    Contains core training settings with nested library-specific configurations.
    Plugins (MLflow, Optuna) are moved to top-level settings.
    Data configuration is moved to top-level data section.

    Args:
        trainer: PyTorch Lightning trainer settings (library-specific - kept nested).
        optimizer: Optimizer and scheduler settings (library-specific - kept nested).
        resume_from_checkpoint: Checkpoint to resume training from (full state).
        lr_tuner: Learning rate tuner settings.
        stopping: Early stopping configuration.
        loss: Loss function configuration.
        metrics: Metrics to compute during training.
    """

    # Library-specific configurations (kept nested as requested)
    trainer: TrainerSettings = Field(
        default_factory=TrainerSettings, description="PyTorch Lightning trainer settings"
    )
    optimizer: OptimizerPolicySettings = Field(
        default_factory=OptimizerPolicySettings, description="Optimizer policy settings"
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

    lr_tuner: LRTunerSettings | None = Field(
        default=None,
        description="Learning rate tuner settings. If configured, automatic LR tuning runs before training.",
    )

    # Early stopping (replaces top-level epochs/patience/monitor_metric/mode)
    stopping: StoppingSettings = Field(
        default_factory=StoppingSettings, description="Early stopping configuration"
    )

    # Loss function and metrics configuration
    loss: LossComponentSettings = Field(
        default_factory=LossComponentSettings,
        description="Loss function settings for training and validation",
    )
    metrics: tuple[MetricComponentSettings, ...] = Field(
        default=tuple(), description="List of metrics to compute on the model at test time"
    )

    @field_validator("loss", mode="before")
    @classmethod
    def _coerce_loss_string(cls, v: object) -> object:
        """Coerce a bare string loss name to a LossComponentSettings dict.

        Args:
            v: Raw field value from TOML (string) or an already-valid dict/model.

        Returns:
            Dict or model suitable for LossComponentSettings validation.
        """
        if isinstance(v, str):
            return {"name": v}
        return v

    @field_validator("optimizer", mode="before")
    @classmethod
    def _coerce_optimizer_flat(cls, v: object) -> object:
        """Coerce a flat optimizer dict (name/lr/…) to OptimizerPolicySettings dict.

        When the TOML provides ``[training.optimizer]`` with ``name``, ``lr``, etc.
        at the top level, wrap it as ``{"default_optimizer": v}`` so that
        ``OptimizerPolicySettings`` validation succeeds.

        Args:
            v: Raw field value from TOML or an already-valid dict/model.

        Returns:
            Dict suitable for OptimizerPolicySettings validation.
        """
        if not isinstance(v, dict):
            return v
        # Already in policy format if it has 'stages' or 'default_optimizer'
        if "stages" in v or "default_optimizer" in v:
            return v
        # Flat optimizer dict: wrap into default_optimizer
        return {"default_optimizer": v}
