"""Tests for batched prediction/evaluation orchestration (batch-iterate/predict/concatenate)."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from dlkit.engine.inference.batch_prediction import run_batched_evaluation, run_batched_prediction


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


def test_run_batched_evaluation_raises_without_predict_target_key(
    make_predictor: Callable[..., MagicMock],
    make_eval_datamodule: Callable[..., MagicMock],
) -> None:
    predictor = make_predictor(predict_target_key="")
    datamodule = make_eval_datamodule(batches=[])

    with pytest.raises(ValueError, match="predict_target_key"):
        run_batched_evaluation(predictor, datamodule)


def test_run_batched_evaluation_raises_when_no_batches(
    make_predictor: Callable[..., MagicMock],
    make_eval_datamodule: Callable[..., MagicMock],
) -> None:
    predictor = make_predictor()
    datamodule = make_eval_datamodule(batches=[])

    with pytest.raises(ValueError, match="no batches"):
        run_batched_evaluation(predictor, datamodule)

    datamodule.setup.assert_called_once_with("test")


def test_run_batched_evaluation_raises_when_target_key_missing(
    make_predictor: Callable[..., MagicMock],
    make_eval_datamodule: Callable[..., MagicMock],
) -> None:
    predictor = make_predictor(predict_target_key="missing")
    datamodule = make_eval_datamodule(
        batches=[({"x": torch.zeros(2, 1)}, {"y": torch.zeros(2, 1)})]
    )

    with pytest.raises(ValueError, match="missing"):
        run_batched_evaluation(predictor, datamodule)


def test_run_batched_evaluation_defaults_to_test_split(
    make_predictor: Callable[..., MagicMock],
    make_eval_datamodule: Callable[..., MagicMock],
) -> None:
    predictor = make_predictor(predictions=[torch.zeros(2, 1)])
    datamodule = make_eval_datamodule(batches=[({"x": torch.zeros(2, 1)}, {"y": torch.ones(2, 1)})])

    predictions, targets = run_batched_evaluation(predictor, datamodule)

    datamodule.setup.assert_called_once_with("test")
    datamodule.test_dataloader.assert_called_once()
    datamodule.predict_dataloader.assert_not_called()
    assert torch.equal(predictions, torch.zeros(2, 1))
    assert torch.equal(targets, torch.ones(2, 1))


def test_run_batched_evaluation_uses_predict_split_when_requested(
    make_predictor: Callable[..., MagicMock],
    make_eval_datamodule: Callable[..., MagicMock],
) -> None:
    predictor = make_predictor(predictions=[torch.zeros(2, 1)])
    datamodule = make_eval_datamodule(batches=[({"x": torch.zeros(2, 1)}, {"y": torch.ones(2, 1)})])

    run_batched_evaluation(predictor, datamodule, split="predict")

    datamodule.setup.assert_called_once_with("predict")
    datamodule.predict_dataloader.assert_called_once()
    datamodule.test_dataloader.assert_not_called()


def test_run_batched_evaluation_concatenates_across_batches(
    make_predictor: Callable[..., MagicMock],
    make_eval_datamodule: Callable[..., MagicMock],
) -> None:
    predictor = make_predictor(predictions=[torch.zeros(2, 1), torch.ones(3, 1)])
    datamodule = make_eval_datamodule(
        batches=[
            ({"x": torch.zeros(2, 1)}, {"y": torch.full((2, 1), 10.0)}),
            ({"x": torch.zeros(3, 1)}, {"y": torch.full((3, 1), 20.0)}),
        ]
    )

    predictions, targets = run_batched_evaluation(predictor, datamodule)

    assert predictions.shape == (5, 1)
    assert torch.equal(
        targets, torch.cat([torch.full((2, 1), 10.0), torch.full((3, 1), 20.0)], dim=0)
    )
