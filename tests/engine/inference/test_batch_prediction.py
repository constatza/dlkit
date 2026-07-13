"""Tests for batched prediction orchestration (batch-iterate/predict/concatenate)."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import torch

from dlkit.engine.inference.batch_prediction import run_batched_prediction


def test_run_batched_prediction_returns_none_without_datamodule(
    make_predictor: Callable[..., MagicMock],
) -> None:
    predictor = make_predictor()

    result = run_batched_prediction(predictor, None)

    assert result is None


def test_run_batched_prediction_returns_none_when_no_batches(
    make_predictor: Callable[..., MagicMock],
    make_datamodule: Callable[..., MagicMock],
) -> None:
    predictor = make_predictor()
    datamodule = make_datamodule(batches=[])

    result = run_batched_prediction(predictor, datamodule)

    assert result is None
    datamodule.setup.assert_called_once_with("predict")


def test_run_batched_prediction_dispatches_named_features_in_order(
    make_predictor: Callable[..., MagicMock],
    make_datamodule: Callable[..., MagicMock],
) -> None:
    predictor = make_predictor(feature_names=("u", "query_coords"))
    datamodule = make_datamodule(
        batches=[{"u": torch.zeros(1, 3), "query_coords": torch.ones(1, 3)}]
    )

    run_batched_prediction(predictor, datamodule)

    _, kwargs = predictor.predict.call_args
    assert tuple(kwargs) == ("u", "query_coords")


def test_run_batched_prediction_falls_back_to_all_feature_keys_without_names(
    make_predictor: Callable[..., MagicMock],
    make_datamodule: Callable[..., MagicMock],
) -> None:
    predictor = make_predictor(feature_names=())
    datamodule = make_datamodule(batches=[{"x": torch.zeros(1, 3)}])

    run_batched_prediction(predictor, datamodule)

    _, kwargs = predictor.predict.call_args
    assert set(kwargs) == {"x"}


def test_run_batched_prediction_concatenates_across_batches(
    make_predictor: Callable[..., MagicMock],
    make_datamodule: Callable[..., MagicMock],
) -> None:
    predictor = make_predictor(
        feature_names=("x",),
        predictions=[torch.zeros(2, 1), torch.ones(3, 1)],
    )
    datamodule = make_datamodule(batches=[{"x": torch.zeros(2, 4)}, {"x": torch.zeros(3, 4)}])

    result = run_batched_prediction(predictor, datamodule)

    assert result is not None
    assert result.shape == (5, 1)
    assert torch.equal(result, torch.cat([torch.zeros(2, 1), torch.ones(3, 1)], dim=0))
