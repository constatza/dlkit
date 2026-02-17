"""Domain models for DLKit API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from lightning.pytorch import LightningDataModule, LightningModule

from dlkit.tools.config import GeneralSettings


@dataclass(frozen=True)
class TrainingResult:
    """Result of a training workflow execution.

    Args:
        model_state: Final model state after training (None if not captured)
        metrics: Training metrics and performance dataflow
        artifacts: Paths to saved artifacts (checkpoints, logs, etc.)
        duration_seconds: Total training time in seconds
        predictions: Raw prediction batches from post-training predict step
    """

    model_state: ModelState | None
    metrics: dict[str, Any]
    artifacts: dict[str, Path]
    duration_seconds: float
    predictions: list[Any] | None = field(default=None)

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

    def _stack_field(self, field: str) -> np.ndarray | None:
        """Concatenate a named field across all prediction batches.

        Handles the ``predict_step`` dict format:
        ``{"predictions": {...}, "targets": {...}, "latents": {...}}``.
        Falls back to direct conversion for plain tensors/arrays (predictions field only).

        Args:
            field: Top-level key to extract ("predictions", "targets", or "latents").

        Returns:
            Concatenated array, or None if no batches or field not present.
        """
        if not self.predictions:
            return None
        batches = []
        for p in self.predictions:
            if isinstance(p, dict):
                inner = p.get(field)
                if inner is None:
                    continue
                if isinstance(inner, dict):
                    if not inner:
                        continue  # empty dict — no data for this field
                    arr = next(iter(inner.values()))
                else:
                    arr = inner
            elif field == "predictions":
                arr = p  # bare tensor/array fallback
            else:
                continue
            batches.append(np.asarray(arr))
        return np.concatenate(batches, axis=0) if batches else None

    def stacked_predictions(self) -> np.ndarray | None:
        """Stack model predictions from all batches into one array.

        Returns:
            Concatenated predictions array, or None if no predictions captured.
        """
        return self._stack_field("predictions")

    def stacked_targets(self) -> np.ndarray | None:
        """Stack ground-truth targets from all batches into one array.

        Targets come from the same data split used for prediction (test split),
        so they align exactly with ``stacked_predictions()``.

        Returns:
            Concatenated targets array, or None if targets not captured.
        """
        return self._stack_field("targets")


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
