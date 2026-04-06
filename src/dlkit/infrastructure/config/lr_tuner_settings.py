"""Learning rate tuner settings for automatic LR finding."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .core.base_settings import BasicSettings


class LRTunerSettings(BasicSettings):
    """Settings for PyTorch Lightning's automatic learning rate tuner.

    The learning rate tuner is enabled by including this section in configuration.
    When enabled, it runs before training to find an optimal initial learning rate
    using the lr_find algorithm. This performs a learning rate range test by training
    with exponentially or linearly increasing learning rates and analyzing the loss curve.

    To enable: Include the [TRAINING.lr_tuner] section (empty section uses all defaults).
    To disable: Omit the [TRAINING.lr_tuner] section entirely.

    Args:
        min_lr: Minimum learning rate to test during range search.
        max_lr: Maximum learning rate to test during range search.
        num_training: Number of learning rate values to test.
        mode: Search strategy for updating learning rate:
            - 'exponential': Increases LR exponentially (recommended)
            - 'linear': Increases LR linearly
        early_stop_threshold: Stop search if loss > early_stop_threshold * best_loss.
            Set to None to disable early stopping.

    Example:
        Enable with all defaults (recommended starting point):
        ```toml
        [TRAINING.lr_tuner]
        ```

        Enable with custom parameters:
        ```toml
        [TRAINING.lr_tuner]
        min_lr = 1e-6
        max_lr = 0.1
        num_training = 100
        mode = "exponential"
        early_stop_threshold = 4.0
        ```

        Disable (omit section entirely):
        ```toml
        # No [TRAINING.lr_tuner] section
        ```
    """

    min_lr: float = Field(default=1e-8, description="Minimum learning rate to investigate")
    max_lr: float = Field(default=1.0, description="Maximum learning rate to investigate")
    num_training: int = Field(default=30, description="Number of learning rates to test")
    mode: Literal["exponential", "linear"] = Field(
        default="exponential", description="Search strategy: 'exponential' or 'linear'"
    )
    early_stop_threshold: float | None = Field(
        default=4.0, description="Stop if loss > threshold * best_loss. None to disable."
    )
