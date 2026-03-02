from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dlkit.core.datatypes.batch import Batch
from dlkit.core.shape_specs.simple_inference import ShapeSummary
from dlkit.runtime.workflows.factories.build_factory import BuildFactory
from dlkit.runtime.workflows.factories.model_detection import ModelType
from dlkit.tools.config.general_settings import GeneralSettings
from dlkit.tools.config.components.model_components import ModelComponentSettings
from dlkit.tools.config.datamodule_settings import DataModuleSettings
from dlkit.tools.config.dataset_settings import DatasetSettings
from dlkit.tools.config.training_settings import TrainingSettings
from dlkit.tools.config.session_settings import SessionSettings
from dlkit.tools.config.core.context import BuildContext
from dlkit.tools.config.core.factories import FactoryProvider


class _FakeDataset:
    def __init__(self, sample: Any):
        self._sample = sample

    def __len__(self) -> int:
        return 100

    def __getitem__(self, idx: int) -> Any:
        return self._sample


class _FakeDataModule:
    pass


class _FakeModel:
    pass


@pytest.fixture
def tmp_checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("dummy")
    return ckpt


def _make_min_settings(sample: Any, *, inference: bool, ckpt: Path | None) -> GeneralSettings:
    # Minimal flattened settings; we will patch factories to avoid importing real components
    ds = DatasetSettings(name="FlexibleDataset", module_path="dlkit.core.datasets")
    dm = DataModuleSettings(name="InMemoryModule", module_path="dlkit.core.datamodules")
    mdl = ModelComponentSettings(name="Dummy", module_path="dlkit.core.models.nn", checkpoint=ckpt)
    tr = TrainingSettings()
    sess = SessionSettings(inference=inference)
    settings = GeneralSettings(SESSION=sess, MODEL=mdl, DATASET=ds, DATAMODULE=dm, TRAINING=tr)
    # Attach a fake dataset sample we want to drive shape inference with
    settings.__dict__["_test_sample"] = sample  # type: ignore[attr-defined]
    return settings


def test_build_factory_flexible_infers_shape_and_uses_wrapper(
    monkeypatch: pytest.MonkeyPatch, tmp_checkpoint: Path
) -> None:
    # Batch-returning dataset so infer_shapes_from_dataset works
    import torch

    batch_sample = Batch(
        features=(torch.zeros(8, 3),),
        targets=(
            torch.ones(
                1,
            ),
        ),
        latents=(),
    )
    settings = _make_min_settings(batch_sample, inference=True, ckpt=tmp_checkpoint)

    # Patch FactoryProvider.create_component to return our fakes
    def _fake_create_component(s, ctx: BuildContext):  # noqa: ANN001
        if s is settings.DATASET:
            return _FakeDataset(settings.__dict__["_test_sample"])  # type: ignore[index]
        if s is settings.DATAMODULE:
            return _FakeDataModule()
        return _FakeModel()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create_component))

    # Force detection to treat the dummy model as shape-aware so shape inference runs
    monkeypatch.setattr(
        "dlkit.runtime.workflows.factories.build_factory.detect_model_type",
        lambda *_: ModelType.SHAPE_AWARE_DLKIT,
    )

    captured_shape_summary: dict[str, Any] = {}

    def _capture_wrapper(*_, **kwargs):  # noqa: ANN001
        captured_shape_summary["summary"] = kwargs.get("shape_summary")
        return _FakeModel()

    # Patch WrapperFactory used inside FlexibleBuildStrategy by targeting the source module
    monkeypatch.setattr(
        "dlkit.runtime.workflows.factories.build_factory.WrapperFactory.create_standard_wrapper",
        staticmethod(_capture_wrapper),
    )

    comps = BuildFactory().build_components(settings)

    assert isinstance(comps.datamodule, _FakeDataModule)
    assert isinstance(comps.model, _FakeModel)
    assert comps.trainer is None  # inference mode
    assert comps.meta.get("dataset_type") == "flexible"
    assert isinstance(comps.shape_spec, ShapeSummary)
    assert comps.shape_spec.in_shapes == ((8, 3),)
    assert comps.shape_spec.out_shapes == ((1,),)
    assert captured_shape_summary["summary"] is comps.shape_spec


def test_build_factory_selects_graph_strategy_and_passes_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_checkpoint: Path
) -> None:
    # Graph-like dict sample; ensure model context receives shape override
    graph_sample = {
        "x": np.zeros((5, 4)),
        "edge_index": np.zeros((2, 8), dtype=int),
        "y": np.ones((5, 1)),
    }
    ds = DatasetSettings(name="Any", module_path="x", type="graph")
    dm = DataModuleSettings()
    mdl = ModelComponentSettings(name="Dummy", module_path="x", checkpoint=tmp_checkpoint)
    settings = GeneralSettings(
        SESSION=SessionSettings(inference=True),
        MODEL=mdl,
        DATASET=ds,
        DATAMODULE=dm,
        TRAINING=TrainingSettings(),
    )

    def _fake_create_component(s, ctx: BuildContext):  # noqa: ANN001
        if s is settings.DATASET:
            return _FakeDataset(graph_sample)
        if s is settings.DATAMODULE:
            return _FakeDataModule()
        return _FakeModel()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create_component))

    monkeypatch.setattr(
        "dlkit.runtime.workflows.factories.build_factory.detect_model_type",
        lambda *_: ModelType.GRAPH,
    )

    def _capture_graph_wrapper(*_, **kwargs):  # noqa: ANN001
        return _FakeModel()

    monkeypatch.setattr(
        "dlkit.runtime.workflows.factories.build_factory.WrapperFactory.create_graph_wrapper",
        staticmethod(_capture_graph_wrapper),
    )

    comps = BuildFactory().build_components(settings)

    assert comps.meta.get("dataset_type") == "graph"
    assert comps.shape_spec is None  # graph strategy returns None shape_spec


def test_build_factory_selects_timeseries_strategy(
    monkeypatch: pytest.MonkeyPatch, tmp_checkpoint: Path
) -> None:
    # Timeseries hint via type; fallback uses flexible inference
    ts_sample = {"x": np.zeros((12, 2)), "y": np.zeros((1,))}
    ds = DatasetSettings(name="Any", module_path="x", type="timeseries")
    dm = DataModuleSettings()
    mdl = ModelComponentSettings(name="Dummy", module_path="x", checkpoint=tmp_checkpoint)
    settings = GeneralSettings(
        SESSION=SessionSettings(inference=True),
        MODEL=mdl,
        DATASET=ds,
        DATAMODULE=dm,
        TRAINING=TrainingSettings(),
    )

    def _fake_create_component(s, ctx: BuildContext):  # noqa: ANN001
        if s is settings.DATASET:
            return _FakeDataset(ts_sample)
        if s is settings.DATAMODULE:
            return _FakeDataModule()
        return _FakeModel()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create_component))

    monkeypatch.setattr(
        "dlkit.runtime.workflows.factories.build_factory.detect_model_type",
        lambda *_: ModelType.TIMESERIES,
    )

    captured_spec: dict[str, Any] = {}

    def _capture_timeseries_wrapper(*_, **kwargs):  # noqa: ANN001
        captured_spec["summary"] = kwargs.get("shape_summary")
        return _FakeModel()

    monkeypatch.setattr(
        "dlkit.runtime.workflows.factories.build_factory.WrapperFactory.create_timeseries_wrapper",
        staticmethod(_capture_timeseries_wrapper),
    )

    comps = BuildFactory().build_components(settings)
    assert comps.meta.get("dataset_type") == "timeseries"
    # Timeseries dataset returns dict sample (not Batch), so shape inference returns None
    assert comps.shape_spec is None
    assert captured_spec["summary"] is None


def test_build_factory_passes_training_optimizer_scheduler_to_wrapper(
    monkeypatch: pytest.MonkeyPatch, tmp_checkpoint: Path
) -> None:
    """Test that optimizer and scheduler settings from TRAINING are passed to WrapperComponentSettings.

    This test verifies the fix for the bug where training optimizer/scheduler configurations
    were being ignored because WrapperComponentSettings() was created with defaults instead
    of using the user-provided settings from settings.TRAINING.
    """
    from dlkit.tools.config.optimizer_settings import OptimizerSettings, SchedulerSettings

    # Create custom optimizer and scheduler settings
    custom_optimizer = OptimizerSettings(name="SGD", lr=0.01)
    custom_scheduler = SchedulerSettings(name="StepLR", factor=0.8, patience=5)

    # Create training settings with custom optimizer/scheduler
    training_settings = TrainingSettings(
        optimizer=custom_optimizer, scheduler=custom_scheduler, epochs=50
    )

    # Create minimal settings using custom training config
    sample = {"x": np.random.randn(10, 5), "y": np.random.randn(10, 3)}
    settings = _make_min_settings(sample, inference=False, ckpt=tmp_checkpoint)
    settings = settings.model_copy(update={"TRAINING": training_settings})

    # Track what wrapper settings are created
    created_wrapper_settings = []

    # Capture the original __init__ method
    from dlkit.tools.config.components.model_components import WrapperComponentSettings

    original_init = WrapperComponentSettings.__init__

    def _capture_wrapper_init(self, **kwargs):
        # Capture the kwargs before calling the original
        created_wrapper_settings.append(kwargs.copy())
        # Call the original __init__ to properly initialize the object
        return original_init(self, **kwargs)

    # Monkey patch to capture wrapper creation
    monkeypatch.setattr(WrapperComponentSettings, "__init__", _capture_wrapper_init)

    def _fake_create_component(s, ctx: BuildContext):  # noqa: ANN001
        if s is settings.DATASET:
            return _FakeDataset(sample)
        if s is settings.DATAMODULE:
            return _FakeDataModule()
        return _FakeModel()

    def _fake_create_wrapper(*args, **kwargs):
        # Return a mock that has the settings we want to verify
        wrapper_mock = types.SimpleNamespace()
        wrapper_settings = kwargs.get("settings")
        wrapper_mock.settings = wrapper_settings
        return wrapper_mock

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create_component))
    from dlkit.core.models.wrappers.factories import WrapperFactory

    monkeypatch.setattr(
        WrapperFactory, "create_standard_wrapper", staticmethod(_fake_create_wrapper)
    )

    # Build components and verify wrapper gets correct settings
    comps = BuildFactory().build_components(settings)

    # Build should produce a components namespace to inspect
    assert hasattr(comps, "model")

    # Verify that at least one wrapper was created with the custom settings
    assert len(created_wrapper_settings) > 0, "No wrapper settings were created"

    # Check that the custom optimizer/scheduler were passed through
    wrapper_kwargs = created_wrapper_settings[0]  # Get first (should be only) wrapper creation

    # Verify optimizer was passed through correctly
    assert "optimizer" in wrapper_kwargs, "Optimizer not passed to wrapper settings"
    passed_optimizer = wrapper_kwargs["optimizer"]
    assert passed_optimizer.name == "SGD", f"Expected SGD optimizer, got {passed_optimizer.name}"
    assert passed_optimizer.lr == 0.01, f"Expected lr=0.01, got {passed_optimizer.lr}"
    # The key test: we're NOT getting defaults (Adam, 1e-3) but our custom values (SGD, 0.01)
    assert passed_optimizer.name != "Adam", "Got default Adam optimizer instead of custom SGD"
    assert passed_optimizer.lr != 1e-3, "Got default lr=1e-3 instead of custom lr=0.01"

    # Verify scheduler was passed through correctly
    assert "scheduler" in wrapper_kwargs, "Scheduler not passed to wrapper settings"
    passed_scheduler = wrapper_kwargs["scheduler"]
    assert passed_scheduler.name == "StepLR", (
        f"Expected StepLR scheduler, got {passed_scheduler.name}"
    )
    assert passed_scheduler.factor == 0.8, f"Expected factor=0.8, got {passed_scheduler.factor}"
    assert passed_scheduler.patience == 5, f"Expected patience=5, got {passed_scheduler.patience}"
    # The key test: we're NOT getting defaults (ReduceLROnPlateau, 0.5, 50) but our custom values
    assert passed_scheduler.name != "ReduceLROnPlateau", (
        "Got default ReduceLROnPlateau instead of custom StepLR"
    )
    assert passed_scheduler.factor != 0.5, "Got default factor=0.5 instead of custom factor=0.8"
    assert passed_scheduler.patience != 50, "Got default patience=50 instead of custom patience=5"


def test_build_factory_handles_none_scheduler_correctly(
    monkeypatch: pytest.MonkeyPatch, tmp_checkpoint: Path
) -> None:
    """Test that None scheduler is handled correctly without causing validation errors."""
    from dlkit.tools.config.optimizer_settings import OptimizerSettings

    # Create optimizer but leave scheduler as None
    custom_optimizer = OptimizerSettings(name="Adam", lr=0.001)

    # Create training settings with scheduler=None
    training_settings = TrainingSettings(
        optimizer=custom_optimizer,
        scheduler=None,  # Explicitly None
        epochs=10,
    )

    sample = {"x": np.random.randn(5, 3), "y": np.random.randn(5, 2)}
    settings = _make_min_settings(sample, inference=False, ckpt=tmp_checkpoint)
    settings = settings.model_copy(update={"TRAINING": training_settings})

    created_wrapper_settings = []

    # Capture the original __init__ method
    from dlkit.tools.config.components.model_components import WrapperComponentSettings

    original_init = WrapperComponentSettings.__init__

    def _capture_wrapper_init(self, **kwargs):
        # Capture the kwargs before calling the original
        created_wrapper_settings.append(kwargs.copy())
        # Call the original __init__ to properly initialize the object
        return original_init(self, **kwargs)

    monkeypatch.setattr(WrapperComponentSettings, "__init__", _capture_wrapper_init)

    def _fake_create_component(s, ctx: BuildContext):  # noqa: ANN001
        if s is settings.DATASET:
            return _FakeDataset(sample)
        if s is settings.DATAMODULE:
            return _FakeDataModule()
        return _FakeModel()

    def _fake_create_wrapper(*args, **kwargs):
        wrapper_mock = types.SimpleNamespace()
        wrapper_mock.settings = kwargs.get("settings")
        return wrapper_mock

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create_component))
    from dlkit.core.models.wrappers.factories import WrapperFactory

    monkeypatch.setattr(
        WrapperFactory, "create_standard_wrapper", staticmethod(_fake_create_wrapper)
    )

    # This should not raise any validation errors
    comps = BuildFactory().build_components(settings)

    assert hasattr(comps, "model")

    # Verify wrapper was created successfully
    assert len(created_wrapper_settings) > 0
    wrapper_kwargs = created_wrapper_settings[0]

    # Should have optimizer but not scheduler in kwargs (since it's None)
    assert "optimizer" in wrapper_kwargs
    passed_optimizer = wrapper_kwargs["optimizer"]
    assert passed_optimizer.name == "Adam", f"Expected Adam optimizer, got {passed_optimizer.name}"
    assert passed_optimizer.lr == 0.001, f"Expected lr=0.001, got {passed_optimizer.lr}"
    # Verify we're using custom values: lr=0.001 (which equals 1e-3) is our custom value, not default
    # The important test is that we got our custom optimizer settings, not defaults
    # Scheduler should not be in kwargs when None (to use default_factory)
    assert "scheduler" not in wrapper_kwargs
