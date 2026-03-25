"""Domain models for DLKit API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from lightning.pytorch import LightningDataModule, LightningModule
from tensordict import TensorDict

from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.workflow_configs import OptimizationWorkflowConfig, TrainingWorkflowConfig
from dlkit.tools.utils.tensordict_utils import NestedKey, tensordict_to_numpy

_STACKED_CACHE_UNSET = object()


@dataclass(frozen=True, slots=True, kw_only=True)
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
    mlflow_run_id: str | None = field(default=None)
    mlflow_tracking_uri: str | None = field(default=None)
    _stacked_cache: TensorDict | object | None = field(
        default=_STACKED_CACHE_UNSET,
        init=False,
        repr=False,
        compare=False,
    )

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

    # Slot-based dataclasses do not expose __dict__, so cache this lazily in
    # an explicit internal slot instead of using functools.cached_property.
    @property
    def stacked(self) -> TensorDict | None:
        """All stacked results, computed lazily and cached.

        Concatenates all per-batch prediction TensorDicts along dim 0 into a
        single TensorDict.  Top-level keys are ``"predictions"``,
        ``"targets"``, and ``"latents"``; ``"targets"`` may itself be a nested
        TensorDict when the model was trained with a multi-entry dataset.

        Use :meth:`to_numpy` to convert all leaf Tensors to numpy arrays while
        preserving the nested key structure.

        Callers detect absent latents via ``stacked["latents"].shape[1] == 0``.

        Returns:
            A :class:`TensorDict` of shape ``(N,)`` where *N* is the total
            number of samples, or ``None`` when no predictions are available.
        """
        if self._stacked_cache is _STACKED_CACHE_UNSET:
            object.__setattr__(self, "_stacked_cache", self._compute_stacked_results())
        return cast(TensorDict | None, self._stacked_cache)

    def to_numpy(self, *keys: NestedKey) -> dict[str, Any] | Any | None:
        """Convert stacked results to CPU numpy arrays.

        Convenience wrapper around :func:`~dlkit.tools.utils.tensordict_to_numpy`
        for the common case of converting ``self.stacked`` directly.  Supports
        both flat top-level keys and nested key paths.

        Args:
            *keys: Optional key names (``str``) or nested key paths
                   (``tuple[str, ...]``) to select before converting.  When
                   omitted all keys are converted.

        Returns:
            Nested ``dict`` of ``np.ndarray`` leaves, or ``None`` if no
            predictions are available.

        Example::

            result.to_numpy()  # everything
            result.to_numpy("predictions")  # flat key
            result.to_numpy(("targets", "y"))  # nested leaf only
            result.to_numpy("predictions", ("targets", "y"))  # mix of flat + nested
        """
        if self.stacked is None:
            return None
        return tensordict_to_numpy(self.stacked, *keys)

    def _compute_stacked_results(self) -> TensorDict | None:
        """Concatenate per-batch prediction TensorDicts along dim 0.

        Returns:
            Stacked :class:`TensorDict` or ``None`` when ``predictions`` is
            empty.
        """
        if not self.predictions:
            return None
        return cast(TensorDict, torch.cat(self.predictions, dim=0))


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceResult:
    """Result of an inference workflow execution.

    Args:
        model_state: Model state used for inference
        predictions: Model predictions
        metrics: Inference metrics if available
        duration_seconds: Total inference time in seconds
    """

    model_state: ModelState | None
    predictions: Any
    metrics: dict[str, Any] | None
    duration_seconds: float


@dataclass(frozen=True, slots=True, kw_only=True)
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
