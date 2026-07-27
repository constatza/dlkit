"""Tests for FitBuildStrategy: the wrapper-free build path for Fittable models."""

from __future__ import annotations

from typing import Any

import pytest

from dlkit.engine.workflows.factories.build_factory import BuildFactory
from dlkit.engine.workflows.factories.dataset_builder import DatasetBuilder
from dlkit.engine.workflows.factories.fit_build_strategy import FitBuildStrategy
from dlkit.infrastructure.config.job_config import FitJobConfig, TrainingJobConfig


class _FakeDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError


class _FakeDataModule:
    pass


class _FakeFittableModel:
    """Stand-in for a real Fittable LightningModule; identity is what's checked."""


def _fit_settings() -> FitJobConfig:
    return FitJobConfig.model_validate(
        {
            "run": {"type": "fit"},
            "model": {"class": "Dummy", "module_path": "dlkit.domain.nn"},
            "data": {"batch_size": 4, "num_workers": 0},
        }
    )


def _training_settings() -> TrainingJobConfig:
    return TrainingJobConfig.model_validate(
        {
            "run": {"type": "train"},
            "model": {"class": "Dummy", "module_path": "dlkit.domain.nn"},
            "data": {"batch_size": 4, "num_workers": 0},
            "training": {"loss": "mse"},
        }
    )


def test_can_handle_accepts_fit_job_config_only() -> None:
    strategy = FitBuildStrategy()

    assert strategy.can_handle(_fit_settings())
    assert not strategy.can_handle(_training_settings())


def test_build_factory_routes_fit_job_config_without_lightning_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _fit_settings()
    fake_model = _FakeFittableModel()

    monkeypatch.setattr(
        DatasetBuilder, "build_dataset", lambda self, s, ctx, overrides: _FakeDataset()
    )
    monkeypatch.setattr(
        DatasetBuilder,
        "build_datamodule",
        lambda self, s, ctx, dataset, split_resolution, *, family=None: _FakeDataModule(),
    )
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.fit_build_strategy._build_model_from_settings",
        lambda model_settings, **kwargs: (fake_model, {}),
    )

    def _fail_if_called(*_: Any, **__: Any) -> Any:
        raise AssertionError("FitBuildStrategy must not route through WrapperFactory")

    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.WrapperFactory.create_standard_wrapper",
        staticmethod(_fail_if_called),
    )
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.WrapperFactory.create_graph_wrapper",
        staticmethod(_fail_if_called),
    )

    comps = BuildFactory().build_components(settings)

    assert comps.model is fake_model
    assert comps.trainer is None
    assert comps.meta.get("dataset_type") == "fit"
