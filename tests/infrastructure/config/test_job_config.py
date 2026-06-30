"""Tests for the new job config settings classes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlkit.infrastructure.config.data_settings import DataSettings
from dlkit.infrastructure.config.job_config import (
    InferenceJobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)
from dlkit.infrastructure.config.model_components import ModelComponentSettings
from dlkit.infrastructure.config.run_settings import RunSettings
from dlkit.infrastructure.config.search_settings import (
    CategoricalParam,
    IntParam,
    SearchSettings,
)
from dlkit.infrastructure.config.training_settings import StoppingSettings, TrainingSettings


@pytest.fixture
def minimal_run() -> dict:
    """Minimal run settings dict."""
    return {"type": "train", "seed": 42}


@pytest.fixture
def minimal_model() -> dict:
    """Minimal model settings dict."""
    return {"class": "ConstantWidthFFNN", "module_path": "dlkit.domain.nn"}


@pytest.fixture
def minimal_data() -> dict:
    """Minimal data settings dict."""
    return {
        "class": "FlexibleDataset",
        "root": "/tmp",
        "batch_size": 32,
        "features": [{"name": "x", "path": "x.npy"}],
        "targets": [{"name": "y", "path": "y.npy"}],
    }


def test_run_settings_type_optional() -> None:
    r = RunSettings.model_validate({})
    assert r.type is None


def test_run_settings_seed_default_none() -> None:
    r = RunSettings.model_validate({"type": "train"})
    assert r.seed is None


def test_model_settings_alias(minimal_model: dict) -> None:
    m = ModelComponentSettings.model_validate(minimal_model)
    assert m.name == "ConstantWidthFFNN"


def test_model_hyperparams_live_directly_under_model() -> None:
    m = ModelComponentSettings.model_validate(
        {"class": "SomeModel", "hidden_size": 128, "custom_kwarg": 99}
    )
    extra = m.model_extra or {}
    assert m.hidden_size == 128
    assert extra["custom_kwarg"] == 99


def test_data_settings_default_batch_size() -> None:
    d = DataSettings.model_validate({})
    assert d.batch_size == 64


def test_data_module_selector_default() -> None:
    d = DataSettings.model_validate({})
    assert d.module.name == "InMemoryModule"


def test_data_settings_alias() -> None:
    d = DataSettings.model_validate({"class": "FlexibleDataset"})
    assert d.name == "FlexibleDataset"


def test_stopping_settings_defaults() -> None:
    s = StoppingSettings.model_validate({})
    assert s.monitor == "val/loss"
    assert s.patience == 10
    assert s.direction == "min"


def test_training_settings_has_stopping() -> None:
    t = TrainingSettings.model_validate({})
    assert isinstance(t.stopping, StoppingSettings)


def test_training_settings_no_epochs_field() -> None:
    """TRAINING.epochs is deleted; trainer.max_epochs is canonical."""
    with pytest.raises(ValidationError):
        TrainingSettings.model_validate({"epochs": 100})


def test_training_settings_loss_not_loss_function() -> None:
    with pytest.raises(ValidationError):
        TrainingSettings.model_validate({"loss_function": {"name": "mse"}})


def test_search_space_discriminated_union() -> None:
    s = SearchSettings.model_validate(
        {
            "space": {
                "training.optimizer.lr": {"type": "log_float", "low": 1e-5, "high": 1e-1},
                "model.num_layers": {"type": "int", "low": 1, "high": 6},
                "model.act": {"type": "categorical", "choices": ["relu", "gelu"]},
            }
        }
    )
    from dlkit.infrastructure.config.search_settings import LogFloatParam

    assert isinstance(s.space["training.optimizer.lr"], LogFloatParam)
    assert isinstance(s.space["model.num_layers"], IntParam)
    assert isinstance(s.space["model.act"], CategoricalParam)
    assert s.space["model.act"].choices == ["relu", "gelu"]


def test_categorical_has_no_low_high() -> None:
    with pytest.raises(ValidationError):
        CategoricalParam.model_validate({"type": "categorical", "low": 0.0, "high": 1.0})


def test_training_job_config_requires_model_data_training(
    minimal_run: dict,
    minimal_model: dict,
    minimal_data: dict,
) -> None:
    with pytest.raises(ValidationError):
        TrainingJobConfig.model_validate({"run": minimal_run})  # missing model/data/training


def test_inference_job_config_accepts_no_data(minimal_run: dict, minimal_model: dict) -> None:
    cfg = InferenceJobConfig.model_validate(
        {
            "run": {**minimal_run, "type": "predict"},
            "model": {**minimal_model, "checkpoint": "/tmp/model.ckpt"},
        }
    )
    assert cfg.model.name == "ConstantWidthFFNN"
    assert cfg.data is None


def test_search_job_config_requires_search_section(
    minimal_run: dict, minimal_model: dict, minimal_data: dict
) -> None:
    with pytest.raises(ValidationError):
        SearchJobConfig.model_validate(
            {
                "run": {**minimal_run, "type": "search"},
                "model": minimal_model,
                "data": minimal_data,
                "training": {},
                # missing search section
            }
        )


def test_model_settings_rejects_both_name_and_class() -> None:
    """Providing both 'name' and 'class' must raise ValidationError."""
    with pytest.raises(ValidationError):
        ModelComponentSettings.model_validate({"name": "FooModel", "class": "BarModel"})


def test_model_settings_class_alias_is_canonical() -> None:
    """'class' key should work as the TOML alias."""
    m = ModelComponentSettings.model_validate({"class": "MyModel"})
    assert m.name == "MyModel"


def test_model_settings_rejects_nested_params_table() -> None:
    with pytest.raises(ValidationError, match="Nested 'model.params' is no longer supported"):
        ModelComponentSettings.model_validate({"class": "MyModel", "params": {"hidden_size": 64}})
