"""Domain models for DLKit API."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from lightning.pytorch import LightningDataModule, LightningModule
from loguru import logger

from dlkit.tools.config import GeneralSettings

# Union type for the result of stacking a single field across batches.
_StackedField = np.ndarray | tuple[np.ndarray, ...] | list | None


def _ensure_tuple(val: Any) -> tuple[Any, ...]:
    """Normalize any value to a tuple of stacking candidates.

    Pure function: no side effects, total (never raises).

    Args:
        val: Any prediction batch value.

    Returns:
        An empty tuple for ``None``, the sequence coerced to tuple for
        list/tuple inputs, or a one-element tuple for everything else.
    """
    match val:
        case None:
            return ()
        case list() | tuple() as seq:
            return tuple(seq)
        case _:
            return (val,)


@dataclass(frozen=True)
class StackedResults:
    """Immutable container for stacked model outputs.

    Each field holds the concatenated result for that output stream across
    all prediction batches.  The type is ``_StackedField`` — a single
    ``np.ndarray`` when the model has one output, a ``tuple`` of arrays for
    multi-output models, a ``list`` for irregular (graph) data, or ``None``
    when the stream was empty.

    Args:
        predictions: Stacked model predictions.
        targets: Stacked ground-truth targets.
        latents: Stacked latent representations.
    """

    predictions: _StackedField = None
    targets: _StackedField = None
    latents: _StackedField = None


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

    # cached_property writes directly to instance.__dict__, bypassing
    # __setattr__, so it is compatible with frozen=True (no __slots__).
    @cached_property
    def stacked(self) -> StackedResults:
        """All stacked results, computed lazily and cached.

        Returns:
            :class:`StackedResults` with predictions, targets, and latents
            concatenated across every prediction batch.
        """
        return self._compute_stacked_results()

    def _compute_stacked_results(self) -> StackedResults:
        """Route raw prediction batches into three parallel streams.

        Single O(N) pass over ``self.predictions``.  Each batch is
        decomposed into ``(p_tuple, t_tuple, l_tuple)`` using structural
        pattern matching and normalised via :func:`_ensure_tuple`.

        Returns:
            :class:`StackedResults` produced by stacking each stream.
        """
        if not self.predictions:
            return StackedResults()

        p_batches: list[tuple[Any, ...]] = []
        t_batches: list[tuple[Any, ...]] = []
        l_batches: list[tuple[Any, ...]] = []

        for p in self.predictions:
            match p:
                case dict() as d:
                    p_batches.append(_ensure_tuple(d.get("predictions")))
                    t_batches.append(_ensure_tuple(d.get("targets")))
                    l_batches.append(_ensure_tuple(d.get("latents")))
                case (preds, targets, latents, *_):
                    p_batches.append(_ensure_tuple(preds))
                    t_batches.append(_ensure_tuple(targets))
                    l_batches.append(_ensure_tuple(latents))
                case (preds, targets):
                    p_batches.append(_ensure_tuple(preds))
                    t_batches.append(_ensure_tuple(targets))
                    l_batches.append(())
                case _:
                    p_batches.append(_ensure_tuple(p))
                    t_batches.append(())
                    l_batches.append(())

        return StackedResults(
            predictions=self._stack_batch_list(p_batches, "predictions"),
            targets=self._stack_batch_list(t_batches, "targets"),
            latents=self._stack_batch_list(l_batches, "latents"),
        )

    def _stack_batch_list(
        self, batches: list[tuple[Any, ...]], field_name: str
    ) -> _StackedField:
        """Concatenate a normalised stream of variable-tuples into arrays.

        Iterates over each positional variable (index ``i``) independently,
        attempting ``np.concatenate`` and falling back to a raw list for
        irregular shapes (e.g. graphs with varying node counts).

        Args:
            batches: List of normalised tuples, one entry per prediction batch.
            field_name: Name used in debug log messages only.

        Returns:
            ``None`` if the stream is empty; a single ``np.ndarray`` for
            single-variable models; a ``tuple`` of arrays for multi-variable
            models; or a mixed tuple/list when some variables are irregular.
        """
        if not batches or not any(batches):
            return None

        num_variables = max(len(b) for b in batches)
        stacked: list[Any] = []

        for i in range(num_variables):
            items = [b[i] for b in batches if i < len(b)]
            if not items:
                continue
            try:
                arrays = [np.asarray(item) for item in items]
                stacked.append(np.concatenate(arrays, axis=0))
            except (ValueError, TypeError):
                logger.debug(
                    f"Field '{field_name}' variable {i} is irregular. "
                    "Returning list of raw items."
                )
                stacked.append(items)

        if not stacked:
            return None
        return stacked[0] if len(stacked) == 1 else tuple(stacked)


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
