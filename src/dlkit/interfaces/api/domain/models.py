"""Domain models for DLKit API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lightning.pytorch import LightningDataModule, LightningModule

from dlkit.tools.config import GeneralSettings


@dataclass(frozen=True)
class TrainingResult:
    """Result of a training workflow execution.

    Args:
        model_state: Final model state after training
        metrics: Training metrics and performance dataflow
        artifacts: Paths to saved artifacts (checkpoints, logs, etc.)
        duration_seconds: Total training time in seconds
    """

    model_state: ModelState
    metrics: dict[str, Any]
    artifacts: dict[str, Path]
    duration_seconds: float

    @property
    def checkpoint_path(self) -> Path | None:
        """Get the checkpoint path from artifacts.

        Returns best checkpoint if available, otherwise last checkpoint.

        Returns:
            Path to checkpoint, or None if no checkpoint available
        """
        if "best_checkpoint" in self.artifacts:
            return self.artifacts["best_checkpoint"]
        if "last_checkpoint" in self.artifacts:
            return self.artifacts["last_checkpoint"]
        return None


@dataclass(frozen=True)
class InferenceResult:
    """Result of an inference workflow execution.

    Args:
        model_state: Model state used for inference
        predictions: Model predictions
        metrics: Inference metrics if available
        duration_seconds: Total inference time in seconds
    """

    model_state: ModelState
    predictions: Any
    metrics: dict[str, Any] | None
    duration_seconds: float


@dataclass(frozen=True)
class OptimizationResult:
    """Result of hyperparameter optimization workflow.

    Args:
        best_trial: Best trial configuration and results
        training_result: Training result with best parameters
        study_summary: Optimization study summary
        duration_seconds: Total optimization time in seconds
    """

    # Best trial can be an Optuna Trial/FrozenTrial or a dict-like view
    best_trial: Any
    training_result: TrainingResult
    study_summary: dict[str, Any]
    duration_seconds: float


@dataclass(frozen=True)
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
    settings: GeneralSettings
