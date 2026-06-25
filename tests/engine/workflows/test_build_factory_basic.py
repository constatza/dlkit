from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from dlkit.engine.workflows.factories.build_factory import BuildFactory
from dlkit.engine.workflows.factories.dataset_builder import DatasetBuilder
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.entry_types import AutoencoderTarget, NpyEntry
from dlkit.infrastructure.config.job_config import TrainingJobConfig
from dlkit.infrastructure.config.model_components import (
    LossComponentSettings,
    LossInputRef,
)


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
def nested_xy_sample() -> TensorDict:
    """Single nested-TensorDict sample with feature ``x`` and target ``y``.

    Returns:
        A ``batch_size=[]`` TensorDict shaped for ``infer_entry_shapes``.
    """
    return TensorDict(
        {
            "features": TensorDict({"x": torch.zeros(2)}, batch_size=[]),
            "targets": TensorDict({"y": torch.zeros(1)}, batch_size=[]),
        },
        batch_size=[],
    )


@pytest.fixture
def tmp_checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("dummy")
    return ckpt


def _make_inference_job(sample: Any, ckpt: Path) -> TrainingJobConfig:
    """Build a TrainingJobConfig in predict mode for strategy tests.

    Using ``run.type = "predict"`` causes ``build_trainer`` to return ``None``
    (inference/predict mode), while still supplying the ``training`` section
    required by ``FlexibleBuildStrategy``.

    Args:
        sample: A fake dataset sample (stored for monkeypatching use only).
        ckpt: Path to the checkpoint file.

    Returns:
        Validated TrainingJobConfig with predict run type.
    """
    job = TrainingJobConfig.model_validate(
        {
            "run": {"type": "predict"},
            "model": {"class": "Dummy", "module_path": "dlkit.domain.nn", "checkpoint": str(ckpt)},
            "data": {"batch_size": 8, "num_workers": 0},
            "training": {"loss": "mse"},
        }
    )
    # Attach a fake dataset sample for test-side monkeypatching
    object.__setattr__(job, "_test_sample", sample)
    return job


def _make_training_job(
    ckpt: Path,
    features: list[dict[str, Any]] | None = None,
    targets: list[dict[str, Any]] | None = None,
    family: str | None = None,
    module_path: str | None = None,
) -> TrainingJobConfig:
    """Build a minimal TrainingJobConfig for strategy tests.

    Args:
        ckpt: Path to the checkpoint file.
        features: Feature entry dicts for the data section.
        targets: Target entry dicts for the data section.
        family: Optional dataset family (e.g. ``"graph"``).
        module_path: Optional dataset module path.

    Returns:
        Validated TrainingJobConfig.
    """
    data_cfg: dict[str, Any] = {"batch_size": 8, "num_workers": 0}
    if features is not None:
        data_cfg["features"] = features
    if targets is not None:
        data_cfg["targets"] = targets
    if family is not None:
        data_cfg["family"] = family
    if module_path is not None:
        data_cfg["module_path"] = module_path

    return TrainingJobConfig.model_validate(
        {
            "run": {"type": "train"},
            "model": {"class": "Dummy", "module_path": "dlkit.domain.nn", "checkpoint": str(ckpt)},
            "data": data_cfg,
            "training": {"loss": "mse"},
        }
    )


def test_build_factory_flexible_uses_contract_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_checkpoint: Path
) -> None:
    # TensorDict-returning dataset so contract inference can sample from it.
    batch_sample = TensorDict(
        {
            "features": TensorDict({"x": torch.zeros(8, 3)}, batch_size=[8]),
            "targets": TensorDict({"y": torch.ones(1)}, batch_size=[]),
        },
        batch_size=[],
    )
    settings = _make_inference_job(batch_sample, tmp_checkpoint)

    # Intercept dataset and datamodule construction at the DatasetBuilder level.
    def _fake_build_flexible_dataset(self, s, ctx, selected_features, selected_targets):
        return _FakeDataset(batch_sample)

    def _fake_build_datamodule(self, s, ctx, dataset, split_resolution, *, family=None):
        return _FakeDataModule()

    monkeypatch.setattr(DatasetBuilder, "build_flexible_dataset", _fake_build_flexible_dataset)
    monkeypatch.setattr(DatasetBuilder, "build_datamodule", _fake_build_datamodule)

    captured_wrapper_kwargs: dict[str, Any] = {}

    def _capture_wrapper(*_, **kwargs):
        captured_wrapper_kwargs.update(kwargs)
        return _FakeModel()

    monkeypatch.setattr(
        "dlkit.engine.adapters.lightning.factories.WrapperFactory.create_standard_wrapper",
        staticmethod(_capture_wrapper),
    )

    comps = BuildFactory().build_components(settings)

    assert isinstance(comps.datamodule, _FakeDataModule)
    assert isinstance(comps.model, _FakeModel)
    assert comps.trainer is None  # inference mode
    assert comps.meta.get("dataset_type") == "flexible"
    # shape_summary must not be passed to create_standard_wrapper.
    assert "shape_summary" not in captured_wrapper_kwargs


def test_build_factory_selects_graph_strategy_and_passes_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_checkpoint: Path
) -> None:
    # Set family="graph" to route through GraphBuildStrategy.
    settings = _make_training_job(tmp_checkpoint, family="graph")

    def _fake_build_dataset_with_tensor_entries(self, s, ctx):
        return _FakeDataset({"x": np.zeros((5, 4)), "edge_index": np.zeros((2, 8), dtype=int)})

    def _fake_build_datamodule(self, s, ctx, dataset, split_resolution, *, family=None):
        return _FakeDataModule()

    monkeypatch.setattr(
        DatasetBuilder, "build_dataset_with_tensor_entries", _fake_build_dataset_with_tensor_entries
    )
    monkeypatch.setattr(DatasetBuilder, "build_datamodule", _fake_build_datamodule)

    def _capture_graph_wrapper(*_, **kwargs):
        return _FakeModel()

    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.WrapperFactory.create_graph_wrapper",
        staticmethod(_capture_graph_wrapper),
    )

    comps = BuildFactory().build_components(settings)

    assert comps.meta.get("dataset_type") == "graph"


def test_build_factory_passes_training_optimizer_scheduler_to_wrapper(
    tmp_checkpoint: Path,
) -> None:
    """When no policy is configured, build_wrapper_components falls back to AdamWSettings."""
    from dlkit.engine.workflows.factories.component_builders import build_wrapper_components
    from dlkit.infrastructure.config.model_components import WrapperComponentSettings
    from dlkit.infrastructure.config.optimizer_component import AdamWSettings

    entry_configs = (
        NpyEntry(name="x", data_role=DataRole.FEATURE),
        NpyEntry(name="y", data_role=DataRole.TARGET),
    )
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

    settings = _make_training_job(
        tmp_checkpoint,
        features=[{"name": "x", "path": str(x_path)}],
        targets=[{"name": "y", "path": str(y_path)}],
    )

    captured: dict[str, Any] = {}

    class _CapturedFlexibleDataset:
        def __init__(self, *, entries):
            captured["features"] = [e for e in entries if e.data_role == DataRole.FEATURE]
            captured["targets"] = [e for e in entries if e.data_role == DataRole.TARGET]
            self._n = 8

        def __len__(self) -> int:
            return self._n

        def __getitem__(self, idx: int) -> TensorDict:
            return TensorDict(
                {
                    "features": TensorDict({"x": torch.zeros(3)}, batch_size=[]),
                    "targets": TensorDict({"y": torch.zeros(1)}, batch_size=[]),
                },
                batch_size=[],
            )

    def _fake_build_datamodule(self, s, ctx, dataset, split_resolution, *, family=None):
        return _FakeDataModule()

    monkeypatch.setattr(
        "dlkit.engine.data.datasets.flexible.FlexibleDataset", _CapturedFlexibleDataset
    )
    monkeypatch.setattr(DatasetBuilder, "build_datamodule", _fake_build_datamodule)
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.WrapperFactory.create_standard_wrapper",
        staticmethod(lambda *_, **__: _FakeModel()),
    )

    comps = BuildFactory().build_components(settings)

    assert isinstance(comps.datamodule, _FakeDataModule)
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
    nested_xy_sample: TensorDict,
) -> None:
    x_path = tmp_path / "x.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.zeros((6, 2), dtype=np.float32))
    np.save(y_path, np.zeros((6, 1), dtype=np.float32))

    settings = _make_training_job(
        tmp_checkpoint,
        features=[{"name": "x", "path": str(x_path)}],
        targets=[{"name": "y", "path": str(y_path)}],
    )

    captured: dict[str, Any] = {}

    def _fake_build_flexible_dataset(self, s, ctx, selected_features, selected_targets):
        captured["features"] = list(selected_features)
        captured["targets"] = list(selected_targets)
        return _FakeDataset(nested_xy_sample)

    def _fake_build_datamodule(self, s, ctx, dataset, split_resolution, *, family=None):
        return _FakeDataModule()

    monkeypatch.setattr(DatasetBuilder, "build_flexible_dataset", _fake_build_flexible_dataset)
    monkeypatch.setattr(DatasetBuilder, "build_datamodule", _fake_build_datamodule)
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
    nested_xy_sample: TensorDict,
) -> None:
    x_path = tmp_path / "x.npy"
    matrix_path = tmp_path / "matrix.npy"
    aux_path = tmp_path / "aux.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.zeros((6, 2), dtype=np.float32))
    np.save(matrix_path, np.zeros((6, 2, 2), dtype=np.float32))
    np.save(aux_path, np.ones((6, 2), dtype=np.float32))
    np.save(y_path, np.zeros((6, 1), dtype=np.float32))

    settings = _make_training_job(
        tmp_checkpoint,
        features=[
            {"name": "x", "path": str(x_path)},
            {"name": "matrix", "path": str(matrix_path), "model_input": False},
            {"name": "aux", "path": str(aux_path), "model_input": False},
        ],
        targets=[{"name": "y", "path": str(y_path)}],
    )

    captured: dict[str, Any] = {}

    def _fake_build_flexible_dataset(self, s, ctx, selected_features, selected_targets):
        captured["dataset_feature_names"] = [e.name for e in selected_features]
        captured["dataset_target_names"] = [e.name for e in selected_targets]
        return _FakeDataset(nested_xy_sample)

    def _fake_build_datamodule(self, s, ctx, dataset, split_resolution, *, family=None):
        return _FakeDataModule()

    def _capture_wrapper(*_, **kwargs):
        captured["entry_config_names"] = [e.name for e in kwargs.get("entry_configs", ())]
        return _FakeModel()

    monkeypatch.setattr(DatasetBuilder, "build_flexible_dataset", _fake_build_flexible_dataset)
    monkeypatch.setattr(DatasetBuilder, "build_datamodule", _fake_build_datamodule)
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
    nested_xy_sample: TensorDict,
) -> None:
    x_path = tmp_path / "x.npy"
    matrix_path = tmp_path / "matrix.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.zeros((5, 2), dtype=np.float32))
    np.save(matrix_path, np.zeros((5, 2, 2), dtype=np.float32))
    np.save(y_path, np.zeros((5, 1), dtype=np.float32))

    from dlkit.infrastructure.config.training_settings import TrainingSettings

    _training_settings = TrainingSettings(
        loss=LossComponentSettings(
            extra_inputs=(LossInputRef(arg="matrix", key="features.matrix"),)
        )
    )

    job = TrainingJobConfig.model_validate(
        {
            "run": {"type": "train"},
            "model": {
                "class": "Dummy",
                "module_path": "dlkit.domain.nn",
                "checkpoint": str(tmp_checkpoint),
            },
            "data": {
                "batch_size": 8,
                "num_workers": 0,
                "features": [
                    {"name": "x", "path": str(x_path)},
                    {"name": "matrix", "path": str(matrix_path), "model_input": False},
                ],
                "targets": [{"name": "y", "path": str(y_path)}],
            },
            "training": {
                "loss": {"extra_inputs": [{"arg": "matrix", "key": "features.matrix"}]},
            },
        }
    )

    captured: dict[str, Any] = {}

    def _fake_build_flexible_dataset(self, s, ctx, selected_features, selected_targets):
        captured["dataset_feature_names"] = [e.name for e in selected_features]
        return _FakeDataset(nested_xy_sample)

    def _fake_build_datamodule(self, s, ctx, dataset, split_resolution, *, family=None):
        return _FakeDataModule()

    monkeypatch.setattr(DatasetBuilder, "build_flexible_dataset", _fake_build_flexible_dataset)
    monkeypatch.setattr(DatasetBuilder, "build_datamodule", _fake_build_datamodule)
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.WrapperFactory.create_standard_wrapper",
        staticmethod(lambda *_, **__: _FakeModel()),
    )

    BuildFactory().build_components(job)

    assert captured["dataset_feature_names"] == ["x", "matrix"]


def test_flexible_build_strategy_keeps_metric_routed_feature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_checkpoint: Path,
    nested_xy_sample: TensorDict,
) -> None:
    x_path = tmp_path / "x.npy"
    matrix_path = tmp_path / "matrix.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.zeros((5, 2), dtype=np.float32))
    np.save(matrix_path, np.zeros((5, 2, 2), dtype=np.float32))
    np.save(y_path, np.zeros((5, 1), dtype=np.float32))

    job = TrainingJobConfig.model_validate(
        {
            "run": {"type": "train"},
            "model": {
                "class": "Dummy",
                "module_path": "dlkit.domain.nn",
                "checkpoint": str(tmp_checkpoint),
            },
            "data": {
                "batch_size": 8,
                "num_workers": 0,
                "features": [
                    {"name": "x", "path": str(x_path)},
                    {"name": "matrix", "path": str(matrix_path), "model_input": False},
                ],
                "targets": [{"name": "y", "path": str(y_path)}],
            },
            "training": {
                "metrics": [{"extra_inputs": [{"arg": "matrix", "key": "features.matrix"}]}],
            },
        }
    )

    captured: dict[str, Any] = {}

    def _fake_build_flexible_dataset(self, s, ctx, selected_features, selected_targets):
        captured["dataset_feature_names"] = [e.name for e in selected_features]
        return _FakeDataset(nested_xy_sample)

    def _fake_build_datamodule(self, s, ctx, dataset, split_resolution, *, family=None):
        return _FakeDataModule()

    monkeypatch.setattr(DatasetBuilder, "build_flexible_dataset", _fake_build_flexible_dataset)
    monkeypatch.setattr(DatasetBuilder, "build_datamodule", _fake_build_datamodule)
    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.WrapperFactory.create_standard_wrapper",
        staticmethod(lambda *_, **__: _FakeModel()),
    )

    BuildFactory().build_components(job)

    assert captured["dataset_feature_names"] == ["x", "matrix"]


def test_flexible_build_strategy_keeps_target_feature_ref_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_checkpoint: Path,
    nested_xy_sample: TensorDict,
) -> None:
    matrix_path = tmp_path / "matrix.npy"
    np.save(matrix_path, np.zeros((4, 2, 2), dtype=np.float32))

    settings = _make_training_job(
        tmp_checkpoint,
        features=[{"name": "matrix", "path": str(matrix_path), "model_input": False}],
        targets=[],
    )
    # Inject AutoencoderTarget (not in AnyEntry union) bypassing Pydantic validation.
    # validate_config_complete is also monkeypatched because AutoencoderTarget has no path
    # and would fail the PathBasedEntry validation guard in the new validator.
    recon_target = AutoencoderTarget(name="recon", feature_ref="matrix", data_role=DataRole.TARGET)
    settings.data.__dict__["targets"] = [recon_target]

    monkeypatch.setattr(
        "dlkit.engine.workflows.factories.build_factory.validate_config_complete",
        lambda _: None,
    )

    captured: dict[str, Any] = {}

    def _fake_build_flexible_dataset(self, s, ctx, selected_features, selected_targets):
        captured["dataset_feature_names"] = [
            e.name for e in selected_features if getattr(e, "data_role", None) == DataRole.FEATURE
        ]
        return _FakeDataset(nested_xy_sample)

    def _fake_build_datamodule(self, s, ctx, dataset, split_resolution, *, family=None):
        return _FakeDataModule()

    monkeypatch.setattr(DatasetBuilder, "build_flexible_dataset", _fake_build_flexible_dataset)
    monkeypatch.setattr(DatasetBuilder, "build_datamodule", _fake_build_datamodule)
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
    from dlkit.infrastructure.config.model_components import WrapperComponentSettings
    from dlkit.infrastructure.config.optimizer_component import AdamSettings
    from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings

    _custom_optimizer = OptimizerPolicySettings(default_optimizer=AdamSettings(lr=0.001))

    job = TrainingJobConfig.model_validate(
        {
            "run": {"type": "train"},
            "model": {
                "class": "Dummy",
                "module_path": "dlkit.domain.nn",
                "checkpoint": str(tmp_checkpoint),
            },
            "data": {"batch_size": 8, "num_workers": 0},
            "training": {
                "optimizer": {"default_optimizer": {"name": "Adam", "lr": 0.001}},
            },
        }
    )

    sample = TensorDict(
        {
            "features": TensorDict({"x": torch.zeros(3)}, batch_size=[]),
            "targets": TensorDict({"y": torch.zeros(2)}, batch_size=[]),
        },
        batch_size=[],
    )

    created_wrapper_settings: list[dict[str, Any]] = []

    original_init = WrapperComponentSettings.__init__

    def _capture_wrapper_init(self, **kwargs):
        created_wrapper_settings.append(kwargs.copy())
        return original_init(self, **kwargs)

    monkeypatch.setattr(WrapperComponentSettings, "__init__", _capture_wrapper_init)

    def _fake_build_flexible_dataset(self, s, ctx, selected_features, selected_targets):
        return _FakeDataset(sample)

    def _fake_build_datamodule(self, s, ctx, dataset, split_resolution, *, family=None):
        return _FakeDataModule()

    def _fake_create_wrapper(*args, **kwargs):
        wrapper_mock = types.SimpleNamespace()
        wrapper_mock.settings = kwargs.get("settings")
        return wrapper_mock

    monkeypatch.setattr(DatasetBuilder, "build_flexible_dataset", _fake_build_flexible_dataset)
    monkeypatch.setattr(DatasetBuilder, "build_datamodule", _fake_build_datamodule)
    from dlkit.engine.adapters.lightning.factories import WrapperFactory

    monkeypatch.setattr(
        WrapperFactory, "create_standard_wrapper", staticmethod(_fake_create_wrapper)
    )

    # This should not raise any validation errors
    comps = BuildFactory().build_components(job)

    assert hasattr(comps, "model")

    # Verify wrapper was created successfully
    assert len(created_wrapper_settings) > 0
    wrapper_kwargs = created_wrapper_settings[0]

    # Should have optimizer but not scheduler in kwargs (since it's None)
    assert "optimizer" in wrapper_kwargs
    passed_optimizer = wrapper_kwargs["optimizer"]
    assert isinstance(passed_optimizer.default_optimizer, AdamSettings)
    assert passed_optimizer.default_optimizer.lr == 0.001
    assert passed_optimizer.default_optimizer.name == "Adam"
    assert "scheduler" not in wrapper_kwargs
