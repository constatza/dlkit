"""Domain state types for DLKit workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lightning.pytorch import LightningDataModule, LightningModule

from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.workflow_configs import OptimizationWorkflowConfig, TrainingWorkflowConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelState:
    """Represents the complete state of a model and its components.

    This is the core domain model that encapsulates everything needed
    to train, test, or run inference with a model.

    Args:
        model: The Lightning module
        datamodule: The Lightning datamodule
        trainer: The Lightning trainer (None for inference)
        settings: The configuration settings used
    """

    model: LightningModule
    datamodule: LightningDataModule
    trainer: Any | None
    settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig
