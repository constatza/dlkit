"""Shared fixtures for ``dlkit.engine.inference`` tests."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from dlkit.engine.inference.config import PredictionOutput


@pytest.fixture
def make_predictor() -> Callable[..., MagicMock]:
    """Factory fixture building a mock ``CheckpointPredictor``.

    Returns:
        Callable accepting ``feature_names`` (dispatch order restored from
        checkpoint metadata) and ``predictions`` (a single tensor reused for
        every call, or a list of tensors consumed one per call via
        ``side_effect``) that returns a configured mock predictor.
    """

    def _make(
        feature_names: tuple[str, ...] = ("x",),
        predictions: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> MagicMock:
        predictor = MagicMock()
        predictor.feature_names = feature_names
        if isinstance(predictions, list):
            predictor.predict.side_effect = [PredictionOutput(predictions=p) for p in predictions]
        else:
            predictor.predict.return_value = PredictionOutput(
                predictions=predictions if predictions is not None else torch.ones(1, 2)
            )
        return predictor

    return _make


@pytest.fixture
def make_datamodule() -> Callable[..., MagicMock]:
    """Factory fixture building a mock datamodule yielding fixed feature batches.

    Returns:
        Callable accepting ``batches`` (a list of feature-name-to-tensor dicts,
        one per yielded batch) that returns a mock datamodule whose
        ``predict_dataloader()`` yields ``{"features": batch}`` per entry.
    """

    def _make(batches: list[dict[str, torch.Tensor]]) -> MagicMock:
        datamodule = MagicMock()
        loader = [{"features": batch} for batch in batches]
        datamodule.predict_dataloader.return_value = loader
        return datamodule

    return _make
