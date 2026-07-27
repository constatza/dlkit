"""Tests for the Fittable structural protocol."""

from __future__ import annotations

from dlkit.engine.training.fittable import Fittable

from .conftest import MeanBufferFittable


class _NotFittable:
    """Deliberately missing is_fitted(): must not satisfy Fittable."""

    def fit(self, dataloader: object) -> None:
        pass


def test_conforming_model_satisfies_fittable(fittable_model: MeanBufferFittable) -> None:
    assert isinstance(fittable_model, Fittable)


def test_object_missing_is_fitted_does_not_satisfy_fittable() -> None:
    assert not isinstance(_NotFittable(), Fittable)


def test_plain_object_does_not_satisfy_fittable() -> None:
    assert not isinstance(object(), Fittable)
