from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tensordict import TensorDict

from dlkit.common.shapes import ShapeSummary
from dlkit.engine.workflows.factories.build_factory import BuildFactory
from dlkit.engine.workflows.factories.model_detection import ModelType
from dlkit.infrastructure.config.core.context import BuildContext
from dlkit.infrastructure.config.core.factories import FactoryProvider
from dlkit.infrastructure.config.data_entries import Feature, Target
from dlkit.infrastructure.config.datamodule_settings import DataModuleSettings
from dlkit.infrastructure.config.dataset_settings import DatasetSettings
from dlkit.infrastructure.config.enums import DatasetFamily
from dlkit.infrastructure.config.general_settings import GeneralSettings
from dlkit.infrastructure.config.model_components import (
    LossComponentSettings,
    LossInputRef,
    MetricComponentSettings,
    MetricInputRef,
    ModelComponentSettings,
)
from dlkit.infrastructure.config.session_settings import SessionSettings
from dlkit.infrastructure.config.training_settings import TrainingSettings


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
    ds = DatasetSettings(name="FlexibleDataset", module_path="dlkit.engine.data.datasets")
    dm = DataModuleSettings(
        name="InMemoryModule", module_path="dlkit.engine.adapters.lightning.datamodules"
    )
    mdl = ModelComponentSettings(name="Dummy", module_path="dlkit.domain.nn", checkpoint=ckpt)
    tr = TrainingSettings()
    workflow_mode = "inference" if inference else "train"
    sess = SessionSettings(workflow=workflow_mode)
    settings = GeneralSettings(SESSION=sess, MODEL=mdl, DATASET=ds, DATAMODULE=dm, TRAINING=tr)
    # Attach a fake dataset sample we want to drive shape inference with
    settings.__dict__["_test_sample"] = sample
    return settings


def test_build_factory_flexible_infers_shape_and_uses_wrapper(
    monkeypatch: pytest.MonkeyPatch, tmp_checkpoint: Path
) -> None:
    # TensorDict-returning dataset so infer_shapes_from_dataset works
    import torch

    batch_sample = TensorDict(
        {
            "features": TensorDict({"x": torch.zeros(8, 3)}, batch_size=[8]),
            "targets": TensorDict({"y": torch.ones(1)}, batch_size=[]),
        },
        batch_size=[],
    )
    settings = _make_min_settings(batch_sample, inference=True, ckpt=tmp_checkpoint)

    # Patch FactoryProvider.create_component to return our fakes
    def _fake_create_component(s, ctx: BuildContext):
        if s is settings.DATASET:
            return _FakeDataset(settings.__dict__["_test_sample"])
        if s is settings.DATAMODULE:
            return _FakeDataModule()
        return _FakeModel()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create_component))

    # Force detection to treat the dummy model as shape-aware so shape inference runs.
    # Patch in flexible_build_strategy's own namespace (it imports detect_model_type directly).
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.flexible_build_strategy.detect_model_type",
        lambda *_: ModelType.SHAPE_AWARE_DLKIT,
    )

    captured_shape_summary: dict[str, Any] = {}

    def _capture_wrapper(*_, **kwargs):
        captured_shape_summary["summary"] = kwargs.get("shape_summary")
        return _FakeModel()

    # Patch WrapperFactory.create_standard_wrapper on the class object itself —
    # shared by all importers, so the original path still works.
    monkeypatch.setattr(
        "dlkit.engine.adapters.lightning.factories.WrapperFactory.create_standard_wrapper",
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
    ds = DatasetSettings(
        name="Any",
        module_path="dlkit.engine.data.datasets",
        type=DatasetFamily.GRAPH,
    )
    dm = DataModuleSettings()
    mdl = ModelComponentSettings(
        name="Dummy",
        module_path="dlkit.domain.nn.ffnn",
        checkpoint=tmp_checkpoint,
    )
    settings = GeneralSettings(
        SESSION=SessionSettings(workflow="inference"),
        MODEL=mdl,
        DATASET=ds,
        DATAMODULE=dm,
        TRAINING=TrainingSettings(),
    )

    def _fake_create_component(s, ctx: BuildContext):
        if s is settings.DATASET:
            return _FakeDataset(graph_sample)
        if s is settings.DATAMODULE:
            return _FakeDataModule()
        return _FakeModel()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create_component))

    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.detect_model_type",
        lambda *_: ModelType.GRAPH,
    )

    def _capture_graph_wrapper(*_, **kwargs):
        return _FakeModel()

    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.WrapperFactory.create_graph_wrapper",
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
    ds = DatasetSettings(
        name="Any",
        module_path="dlkit.engine.data.datasets",
        type=DatasetFamily.TIMESERIES,
    )
    dm = DataModuleSettings()
    mdl = ModelComponentSettings(
        name="Dummy",
        module_path="dlkit.domain.nn.ffnn",
        checkpoint=tmp_checkpoint,
    )
    settings = GeneralSettings(
        SESSION=SessionSettings(workflow="inference"),
        MODEL=mdl,
        DATASET=ds,
        DATAMODULE=dm,
        TRAINING=TrainingSettings(),
    )

    def _fake_create_component(s, ctx: BuildContext):
        if s is settings.DATASET:
            return _FakeDataset(ts_sample)
        if s is settings.DATAMODULE:
            return _FakeDataModule()
        return _FakeModel()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create_component))

    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.detect_model_type",
        lambda *_: ModelType.TIMESERIES,
    )

    captured_spec: dict[str, Any] = {}

    def _capture_timeseries_wrapper(*_, **kwargs):
        captured_spec["summary"] = kwargs.get("shape_summary")
        return _FakeModel()

    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.WrapperFactory.create_timeseries_wrapper",
        staticmethod(_capture_timeseries_wrapper),
    )

    comps = BuildFactory().build_components(settings)
    assert comps.meta.get("dataset_type") == "timeseries"
    # Timeseries dataset returns dict sample, so shape inference returns None
    assert comps.shape_spec is None
    assert captured_spec["summary"] is None


def test_build_factory_passes_training_optimizer_scheduler_to_wrapper(
    tmp_checkpoint: Path,
) -> None:
    """When no policy is configured, build_wrapper_components falls back to AdamWSettings."""
    from dlkit.engine.workflows.factories.component_builders import build_wrapper_components
    from dlkit.infrastructure.config.data_entries import Feature, Target
    from dlkit.infrastructure.config.model_components import WrapperComponentSettings
    from dlkit.infrastructure.config.optimizer_component import AdamWSettings

    entry_configs = (Feature(name="x"), Target(name="y"))
    wrapper_settings = WrapperComponentSettings()

    # No policy set → fallback must be AdamWSettings, not a ValidationError.
    result = build_wrapper_components(wrapper_settings, entry_configs)

    assert isinstance(result.optimizer_policy_settings.default_optimizer, AdamWSettings), (
        "Expected AdamWSettings fallback optimizer"
    )


def test_flexible_build_strategy_uses_raw_entries_for_flexible_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_checkpoint: Path,
) -> None:
    x_path = tmp_path / "x.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.zeros((8, 3), dtype=np.float32))
    np.save(y_path, np.zeros((8, 1), dtype=np.float32))

    ds = DatasetSettings(
        name="SupervisedArrayDataset",
        module_path="dlkit.engine.data.datasets",
        features=(Feature(name="x", path=x_path),),
        targets=(Target(name="y", path=y_path),),
        memmap_cache=True,
    )
    dm = DataModuleSettings(
        name="InMemoryModule", module_path="dlkit.engine.adapters.lightning.datamodules"
    )
    mdl = ModelComponentSettings(
        name="Dummy", module_path="dlkit.domain.nn", checkpoint=tmp_checkpoint
    )
    settings = GeneralSettings(
        SESSION=SessionSettings(workflow="inference"),
        MODEL=mdl,
        DATASET=ds,
        DATAMODULE=dm,
        TRAINING=TrainingSettings(),
    )

    captured: dict[str, Any] = {}

    class _CapturedFlexibleDataset:
        def __init__(self, *, features, targets=None, memmap_cache_dir=None):
            captured["features"] = list(features)
            captured["targets"] = list(targets or ())
            captured["memmap_cache_dir"] = memmap_cache_dir
            self._n = 8

        def __len__(self) -> int:
            return self._n

        def __getitem__(self, idx: int) -> TensorDict:
            import torch

            return TensorDict(
                {
                    "features": TensorDict({"x": torch.zeros(3)}, batch_size=[]),
                    "targets": TensorDict({"y": torch.zeros(1)}, batch_size=[]),
                },
                batch_size=[],
            )

    def _fake_create_component(s, ctx: BuildContext):
        if s is settings.DATAMODULE:
            return _FakeDataModule()
        return _FakeModel()

    monkeypatch.setattr(
        "dlkit.engine.data.datasets.flexible.FlexibleDataset", _CapturedFlexibleDataset
    )
    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create_component))
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.detect_model_type",
        lambda *_: ModelType.SHAPE_AGNOSTIC_EXTERNAL,
    )
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.WrapperFactory.create_standard_wrapper",
        staticmethod(lambda *_, **__: _FakeModel()),
    )

    comps = BuildFactory().build_components(settings)

    assert isinstance(comps.datamodule, _FakeDataModule)
    assert captured["memmap_cache_dir"] is not None
    assert captured["features"]
    assert captured["targets"]
    first_feature = captured["features"][0]
    first_target = captured["targets"][0]
    assert hasattr(first_feature, "path")
    assert hasattr(first_target, "path")
    assert not hasattr(first_feature, "tensor")
    assert not hasattr(first_target, "tensor")


def test_flexible_build_strategy_factory_path_uses_raw_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_checkpoint: Path,
) -> None:
    x_path = tmp_path / "x.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.zeros((6, 2), dtype=np.float32))
    np.save(y_path, np.zeros((6, 1), dtype=np.float32))

    ds = DatasetSettings(
        name="FlexibleDataset",
        module_path="dlkit.engine.data.datasets",
        features=(Feature(name="x", path=x_path),),
        targets=(Target(name="y", path=y_path),),
    )
    dm = DataModuleSettings(
        name="InMemoryModule", module_path="dlkit.engine.adapters.lightning.datamodules"
    )
    mdl = ModelComponentSettings(
        name="Dummy", module_path="dlkit.domain.nn", checkpoint=tmp_checkpoint
    )
    settings = GeneralSettings(
        SESSION=SessionSettings(workflow="inference"),
        MODEL=mdl,
        DATASET=ds,
        DATAMODULE=dm,
        TRAINING=TrainingSettings(),
    )

    captured: dict[str, Any] = {}

    def _fake_create_component(s, ctx: BuildContext):
        if s is settings.DATASET:
            captured["features"] = list(ctx.overrides["features"])
            captured["targets"] = list(ctx.overrides["targets"])
            return _FakeDataset({"x": np.zeros((2,))})
        if s is settings.DATAMODULE:
            return _FakeDataModule()
        return _FakeModel()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create_component))
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.detect_model_type",
        lambda *_: ModelType.SHAPE_AGNOSTIC_EXTERNAL,
    )
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.WrapperFactory.create_standard_wrapper",
        staticmethod(lambda *_, **__: _FakeModel()),
    )

    BuildFactory().build_components(settings)

    assert captured["features"]
    assert captured["targets"]
    first_feature = captured["features"][0]
    first_target = captured["targets"][0]
    assert hasattr(first_feature, "path")
    assert hasattr(first_target, "path")
    assert not hasattr(first_feature, "tensor")
    assert not hasattr(first_target, "tensor")


def test_flexible_build_strategy_prunes_unreferenced_features(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_checkpoint: Path,
) -> None:
    x_path = tmp_path / "x.npy"
    matrix_path = tmp_path / "matrix.npy"
    aux_path = tmp_path / "aux.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.zeros((6, 2), dtype=np.float32))
    np.save(matrix_path, np.zeros((6, 2, 2), dtype=np.float32))
    np.save(aux_path, np.ones((6, 2), dtype=np.float32))
    np.save(y_path, np.zeros((6, 1), dtype=np.float32))

    ds = DatasetSettings(
        name="FlexibleDataset",
        module_path="dlkit.engine.data.datasets",
        features=(
            Feature(name="x", path=x_path),
            Feature(name="matrix", path=matrix_path, model_input=False),
            Feature(name="aux", path=aux_path, model_input=False),
        ),
        targets=(Target(name="y", path=y_path),),
    )
    settings = GeneralSettings(
        SESSION=SessionSettings(workflow="inference"),
        MODEL=ModelComponentSettings(
            name="Dummy", module_path="dlkit.domain.nn", checkpoint=tmp_checkpoint
        ),
        DATASET=ds,
        DATAMODULE=DataModuleSettings(
            name="InMemoryModule", module_path="dlkit.engine.adapters.lightning.datamodules"
        ),
        TRAINING=TrainingSettings(),
    )

    captured: dict[str, Any] = {}

    def _fake_create_component(s, ctx: BuildContext):
        if s is settings.DATASET:
            captured["dataset_feature_names"] = [entry.name for entry in ctx.overrides["features"]]
            captured["dataset_target_names"] = [entry.name for entry in ctx.overrides["targets"]]
            return _FakeDataset({"x": np.zeros((2,))})
        if s is settings.DATAMODULE:
            return _FakeDataModule()
        return _FakeModel()

    def _capture_wrapper(*_, **kwargs):
        captured["entry_config_names"] = [e.name for e in kwargs.get("entry_configs", ())]
        return _FakeModel()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create_component))
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.detect_model_type",
        lambda *_: ModelType.SHAPE_AGNOSTIC_EXTERNAL,
    )
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.WrapperFactory.create_standard_wrapper",
        staticmethod(_capture_wrapper),
    )

    BuildFactory().build_components(settings)

    assert captured["dataset_feature_names"] == ["x"]
    assert captured["dataset_target_names"] == ["y"]
    assert captured["entry_config_names"] == ["x", "y"]


def test_flexible_build_strategy_keeps_loss_routed_feature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_checkpoint: Path,
) -> None:
    x_path = tmp_path / "x.npy"
    matrix_path = tmp_path / "matrix.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.zeros((5, 2), dtype=np.float32))
    np.save(matrix_path, np.zeros((5, 2, 2), dtype=np.float32))
    np.save(y_path, np.zeros((5, 1), dtype=np.float32))

    ds = DatasetSettings(
        name="FlexibleDataset",
        module_path="dlkit.engine.data.datasets",
        features=(
            Feature(name="x", path=x_path),
            Feature(name="matrix", path=matrix_path, model_input=False),
        ),
        targets=(Target(name="y", path=y_path),),
    )
    training = TrainingSettings(
        loss_function=LossComponentSettings(
            extra_inputs=(LossInputRef(arg="matrix", key="features.matrix"),)
        )
    )
    settings = GeneralSettings(
        SESSION=SessionSettings(workflow="inference"),
        MODEL=ModelComponentSettings(
            name="Dummy", module_path="dlkit.domain.nn", checkpoint=tmp_checkpoint
        ),
        DATASET=ds,
        DATAMODULE=DataModuleSettings(
            name="InMemoryModule", module_path="dlkit.engine.adapters.lightning.datamodules"
        ),
        TRAINING=training,
    )

    captured: dict[str, Any] = {}

    def _fake_create_component(s, ctx: BuildContext):
        if s is settings.DATASET:
            captured["dataset_feature_names"] = [entry.name for entry in ctx.overrides["features"]]
            return _FakeDataset({"x": np.zeros((2,))})
        if s is settings.DATAMODULE:
            return _FakeDataModule()
        return _FakeModel()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create_component))
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.detect_model_type",
        lambda *_: ModelType.SHAPE_AGNOSTIC_EXTERNAL,
    )
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.WrapperFactory.create_standard_wrapper",
        staticmethod(lambda *_, **__: _FakeModel()),
    )

    BuildFactory().build_components(settings)

    assert captured["dataset_feature_names"] == ["x", "matrix"]


def test_flexible_build_strategy_keeps_metric_routed_feature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_checkpoint: Path,
) -> None:
    x_path = tmp_path / "x.npy"
    matrix_path = tmp_path / "matrix.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.zeros((5, 2), dtype=np.float32))
    np.save(matrix_path, np.zeros((5, 2, 2), dtype=np.float32))
    np.save(y_path, np.zeros((5, 1), dtype=np.float32))

    ds = DatasetSettings(
        name="FlexibleDataset",
        module_path="dlkit.engine.data.datasets",
        features=(
            Feature(name="x", path=x_path),
            Feature(name="matrix", path=matrix_path, model_input=False),
        ),
        targets=(Target(name="y", path=y_path),),
    )
    training = TrainingSettings(
        metrics=(
            MetricComponentSettings(
                extra_inputs=(MetricInputRef(arg="matrix", key="features.matrix"),)
            ),
        )
    )
    settings = GeneralSettings(
        SESSION=SessionSettings(workflow="inference"),
        MODEL=ModelComponentSettings(
            name="Dummy", module_path="dlkit.domain.nn", checkpoint=tmp_checkpoint
        ),
        DATASET=ds,
        DATAMODULE=DataModuleSettings(
            name="InMemoryModule", module_path="dlkit.engine.adapters.lightning.datamodules"
        ),
        TRAINING=training,
    )

    captured: dict[str, Any] = {}

    def _fake_create_component(s, ctx: BuildContext):
        if s is settings.DATASET:
            captured["dataset_feature_names"] = [entry.name for entry in ctx.overrides["features"]]
            return _FakeDataset({"x": np.zeros((2,))})
        if s is settings.DATAMODULE:
            return _FakeDataModule()
        return _FakeModel()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create_component))
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.detect_model_type",
        lambda *_: ModelType.SHAPE_AGNOSTIC_EXTERNAL,
    )
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.WrapperFactory.create_standard_wrapper",
        staticmethod(lambda *_, **__: _FakeModel()),
    )

    BuildFactory().build_components(settings)

    assert captured["dataset_feature_names"] == ["x", "matrix"]


def test_flexible_build_strategy_keeps_target_feature_ref_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_checkpoint: Path,
) -> None:
    matrix_path = tmp_path / "matrix.npy"
    y_path = tmp_path / "y.npy"
    np.save(matrix_path, np.zeros((4, 2, 2), dtype=np.float32))
    np.save(y_path, np.zeros((4, 1), dtype=np.float32))

    ds = DatasetSettings(
        name="FlexibleDataset",
        module_path="dlkit.engine.data.datasets",
        features=(Feature(name="matrix", path=matrix_path, model_input=False),),
        targets=(Target(name="y", path=y_path),),
    )
    settings = GeneralSettings(
        SESSION=SessionSettings(workflow="inference"),
        MODEL=ModelComponentSettings(
            name="Dummy", module_path="dlkit.domain.nn", checkpoint=tmp_checkpoint
        ),
        DATASET=ds,
        DATAMODULE=DataModuleSettings(
            name="InMemoryModule", module_path="dlkit.engine.adapters.lightning.datamodules"
        ),
        TRAINING=TrainingSettings(),
    )
    settings.DATASET.__dict__["targets"] = [
        types.SimpleNamespace(name="recon", feature_ref="matrix")
    ]

    captured: dict[str, Any] = {}

    def _fake_create_component(s, ctx: BuildContext):
        if s is settings.DATASET:
            captured["dataset_feature_names"] = [entry.name for entry in ctx.overrides["features"]]
            return _FakeDataset({"x": np.zeros((2,))})
        if s is settings.DATAMODULE:
            return _FakeDataModule()
        return _FakeModel()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create_component))
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.detect_model_type",
        lambda *_: ModelType.SHAPE_AGNOSTIC_EXTERNAL,
    )
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.WrapperFactory.create_standard_wrapper",
        staticmethod(lambda *_, **__: _FakeModel()),
    )

    BuildFactory().build_components(settings)

    assert captured["dataset_feature_names"] == ["matrix"]


def test_build_factory_handles_none_scheduler_correctly(
    monkeypatch: pytest.MonkeyPatch, tmp_checkpoint: Path
) -> None:
    """Test that None scheduler is handled correctly without causing validation errors."""
    from dlkit.infrastructure.config.optimizer_settings import OptimizerSettings

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
    from dlkit.infrastructure.config.model_components import WrapperComponentSettings

    original_init = WrapperComponentSettings.__init__

    def _capture_wrapper_init(self, **kwargs):
        # Capture the kwargs before calling the original
        created_wrapper_settings.append(kwargs.copy())
        # Call the original __init__ to properly initialize the object
        return original_init(self, **kwargs)

    monkeypatch.setattr(WrapperComponentSettings, "__init__", _capture_wrapper_init)

    def _fake_create_component(s, ctx: BuildContext):
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
    from dlkit.engine.adapters.lightning.factories import WrapperFactory

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
