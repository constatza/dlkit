"""Unit tests for the build-phase transform-fitting trigger.

fit_transforms_if_needed() is the single call site that fits a model's batch
transformer before any Trainer/Tuner exists (see IBuildStrategy.build()).
"""

from __future__ import annotations

from typing import cast
from unittest.mock import Mock

from dlkit.engine.training.transform_fitting import (
    IFittableTransformer,
    IHasTrainDataloader,
    fit_if_needed,
    fit_transforms_if_needed,
)


def _fittable_transformer(*, fitted: bool) -> Mock:
    transformer = Mock(spec=IFittableTransformer)
    transformer.is_fitted.return_value = fitted
    return transformer


def test_fit_if_needed_fits_an_unfitted_transformer() -> None:
    transformer = _fittable_transformer(fitted=False)

    fit_if_needed(transformer, "the-dataloader")

    transformer.fit.assert_called_once_with("the-dataloader")


def test_fit_if_needed_does_not_refit_an_already_fitted_transformer() -> None:
    transformer = _fittable_transformer(fitted=True)

    fit_if_needed(transformer, "the-dataloader")

    transformer.fit.assert_not_called()


def test_fit_if_needed_is_a_noop_for_non_fittable_objects() -> None:
    # Deliberately not IFittableTransformer-shaped — exercises the runtime
    # isinstance guard; cast documents the intentional type mismatch.
    not_fittable = cast(IFittableTransformer, object())

    fit_if_needed(not_fittable, "the-dataloader")  # must not raise


def test_fit_transforms_if_needed_fits_model_batch_transformer_from_datamodule() -> None:
    transformer = _fittable_transformer(fitted=False)
    model = Mock()
    model.batch_transformer = transformer
    datamodule = Mock(spec=IHasTrainDataloader)
    datamodule.train_dataloader.return_value = "the-train-loader"

    fit_transforms_if_needed(model, datamodule)

    transformer.fit.assert_called_once_with("the-train-loader")


def test_fit_transforms_if_needed_is_a_noop_when_model_has_no_batch_transformer() -> None:
    model = object()
    datamodule = Mock(spec=IHasTrainDataloader)
    datamodule.train_dataloader.return_value = "the-train-loader"

    fit_transforms_if_needed(model, datamodule)  # must not raise

    datamodule.train_dataloader.assert_not_called()


def test_fit_transforms_if_needed_is_a_noop_when_datamodule_has_no_train_dataloader() -> None:
    transformer = _fittable_transformer(fitted=False)
    model = Mock()
    model.batch_transformer = transformer
    datamodule = object()

    fit_transforms_if_needed(model, datamodule)  # must not raise

    transformer.fit.assert_not_called()
