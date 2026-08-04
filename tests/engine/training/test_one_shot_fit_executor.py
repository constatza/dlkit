"""Tests for OneShotFitExecutor: the trainer-free, closed-form fit path."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch.utils.data import DataLoader

from dlkit.common.errors import WorkflowError
from dlkit.engine.artifacts import read_checkpoint_artifacts
from dlkit.engine.inference.checkpoint_reader import extract_model_settings
from dlkit.engine.inference.model_builder import build_model_from_checkpoint
from dlkit.engine.training.components import RuntimeComponents
from dlkit.engine.training.one_shot_fit_executor import OneShotFitExecutor
from dlkit.infrastructure.config.job_config import FitJobConfig
from dlkit.infrastructure.io.path_context import path_override_context

from .conftest import LazyBufferFittable, MeanBufferFittable


@pytest.fixture
def fit_settings() -> FitJobConfig:
    return FitJobConfig.model_validate(
        {
            "run": {"type": "fit", "seed": 0},
            "model": {
                "class": "MeanBufferFittable",
                "module_path": "tests.engine.training.conftest",
            },
            "data": {"batch_size": 4, "num_workers": 0},
        }
    )


class _StubDataModule:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def train_dataloader(self) -> DataLoader:
        return self._loader


@pytest.fixture
def components(fittable_model: MeanBufferFittable, fit_dataloader: DataLoader) -> RuntimeComponents:
    return RuntimeComponents(
        model=fittable_model,
        datamodule=cast(Any, _StubDataModule(fit_dataloader)),
        trainer=None,
        meta={},
    )


def test_execute_calls_fit_exactly_once(
    components: RuntimeComponents, fit_settings: FitJobConfig, tmp_path: Path
) -> None:
    with path_override_context({"root_dir": tmp_path}):
        OneShotFitExecutor().execute(components, fit_settings)

    model = cast(MeanBufferFittable, components.model)
    assert model.fit_call_count == 1
    assert model.is_fitted()
    assert torch.equal(model.mean, torch.tensor([2.5]))


def test_execute_writes_checkpoint_file_under_resolved_root(
    components: RuntimeComponents, fit_settings: FitJobConfig, tmp_path: Path
) -> None:
    with path_override_context({"root_dir": tmp_path}):
        OneShotFitExecutor().execute(components, fit_settings)

    checkpoint_files = list(tmp_path.rglob("fitted.ckpt"))
    assert len(checkpoint_files) == 1
    checkpoint = torch.load(checkpoint_files[0], weights_only=False)
    # No "model." prefix: a Fittable model IS the top-level LightningModule,
    # unlike CoreLightningWrapper-wrapped models (see _build_checkpoint).
    assert "mean" in checkpoint["state_dict"]
    assert torch.equal(checkpoint["state_dict"]["mean"], torch.tensor([2.5]))


def test_execute_attaches_checkpoint_artifact_to_model(
    components: RuntimeComponents, fit_settings: FitJobConfig, tmp_path: Path
) -> None:
    with path_override_context({"root_dir": tmp_path}):
        OneShotFitExecutor().execute(components, fit_settings)

    artifacts = read_checkpoint_artifacts(components.model)
    assert len(artifacts) == 1
    assert artifacts[0].kind == "checkpoint"


def test_execute_checkpoint_model_settings_round_trip_through_extract_model_settings(
    components: RuntimeComponents, fit_settings: FitJobConfig, tmp_path: Path
) -> None:
    with path_override_context({"root_dir": tmp_path}):
        OneShotFitExecutor().execute(components, fit_settings)

    checkpoint_files = list(tmp_path.rglob("fitted.ckpt"))
    checkpoint = torch.load(checkpoint_files[0], weights_only=False)

    reconstructed = extract_model_settings(checkpoint)

    assert reconstructed.name == "MeanBufferFittable"
    assert reconstructed.module_path == "tests.engine.training.conftest"


def test_execute_checkpoint_reconstructs_a_working_model_via_build_model_from_checkpoint(
    components: RuntimeComponents, fit_settings: FitJobConfig, tmp_path: Path
) -> None:
    """The actual claim the feature rests on: build_model_from_checkpoint()
    (real importlib resolution + load_state_dict), not just metadata parsing."""
    with path_override_context({"root_dir": tmp_path}):
        OneShotFitExecutor().execute(components, fit_settings)

    checkpoint_files = list(tmp_path.rglob("fitted.ckpt"))
    checkpoint = torch.load(checkpoint_files[0], weights_only=False)

    rebuilt = build_model_from_checkpoint(checkpoint)

    assert isinstance(rebuilt, MeanBufferFittable)
    assert torch.equal(rebuilt.mean, torch.tensor([2.5]))


@pytest.fixture
def lazy_fit_settings() -> FitJobConfig:
    return FitJobConfig.model_validate(
        {
            "run": {"type": "fit", "seed": 0},
            "model": {
                "class": "LazyBufferFittable",
                "module_path": "tests.engine.training.conftest",
                "rank": 2,
            },
            "data": {"batch_size": 4, "num_workers": 0},
        }
    )


@pytest.fixture
def lazy_components(
    lazy_buffer_model: LazyBufferFittable, lazy_buffer_dataloader: DataLoader
) -> RuntimeComponents:
    return RuntimeComponents(
        model=lazy_buffer_model,
        datamodule=cast(Any, _StubDataModule(lazy_buffer_dataloader)),
        trainer=None,
        meta={},
    )


def test_execute_reconstructs_a_model_whose_buffer_is_registered_only_in_fit(
    lazy_components: RuntimeComponents, lazy_fit_settings: FitJobConfig, tmp_path: Path
) -> None:
    """Regression test for a gap MeanBufferFittable's fixed (1,)-shaped
    buffer can't catch: a Fittable (like torchalg's PODCoarseningStrategy)
    that registers no buffer at all in __init__, only inside fit(), at a
    shape resolved from real data. build_model_from_checkpoint() must
    pre-register such checkpoint-only keys before load_state_dict(strict=True),
    or a freshly-constructed instance rejects the checkpoint with
    "Unexpected key(s) in state_dict"."""
    with path_override_context({"root_dir": tmp_path}):
        OneShotFitExecutor().execute(lazy_components, lazy_fit_settings)

    checkpoint_files = list(tmp_path.rglob("fitted.ckpt"))
    checkpoint = torch.load(checkpoint_files[0], weights_only=False)

    rebuilt = build_model_from_checkpoint(checkpoint)

    assert isinstance(rebuilt, LazyBufferFittable)
    assert rebuilt.is_fitted()
    original = cast(LazyBufferFittable, lazy_components.model)
    assert torch.equal(cast(torch.Tensor, rebuilt.basis), cast(torch.Tensor, original.basis))


def test_execute_raises_when_model_not_fittable(
    fit_settings: FitJobConfig, fit_dataloader: DataLoader
) -> None:
    components = RuntimeComponents(
        model=cast(Any, object()),
        datamodule=cast(Any, _StubDataModule(fit_dataloader)),
        trainer=None,
        meta={},
    )

    with pytest.raises(WorkflowError, match="does not implement Fittable"):
        OneShotFitExecutor().execute(components, fit_settings)


def test_execute_wraps_fit_exception_in_workflow_error(
    fit_settings: FitJobConfig, fit_dataloader: DataLoader
) -> None:
    class _BrokenFittable(MeanBufferFittable):
        def fit(self, dataloader: object) -> None:
            raise RuntimeError("closed-form fit blew up")

    components = RuntimeComponents(
        model=_BrokenFittable(),
        datamodule=cast(Any, _StubDataModule(fit_dataloader)),
        trainer=None,
        meta={},
    )

    with pytest.raises(WorkflowError, match="One-shot fit failed"):
        OneShotFitExecutor().execute(components, fit_settings)
