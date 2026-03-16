"""Test checkpoint persistence in ProcessingLightningWrapper."""

import pytest
import torch
from pathlib import Path
from torch import nn

from dlkit.core.models.wrappers.standard import StandardLightningWrapper
from dlkit.tools.config import (
    WrapperComponentSettings,
    ModelComponentSettings,
)
from dlkit.tools.config.core.context import BuildContext
from dlkit.tools.config.core.factories import FactoryProvider
from dlkit.core.shape_specs import create_shape_spec
from dlkit.tools.config.data_entries import Feature


class _DummyModel(nn.Module):
    """Minimal model for testing."""

    def forward(self, x):
        return x


@pytest.fixture
def patch_factory(monkeypatch):
    """Mock factory to avoid complex config requirements."""

    def _fake_create(settings, ctx: BuildContext):
        if isinstance(settings, ModelComponentSettings):
            return _DummyModel()
        return lambda a, b: torch.nn.functional.mse_loss(a, b)

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create))


@pytest.fixture
def dummy_model_settings() -> ModelComponentSettings:
    """Create minimal model settings for testing."""
    return ModelComponentSettings(
        name="_DummyModel",
        module_path="tests.core.models.wrappers.test_checkpoint_version_validation",
    )


@pytest.fixture
def dummy_wrapper_settings() -> WrapperComponentSettings:
    """Create minimal wrapper settings for testing."""
    return WrapperComponentSettings()


@pytest.fixture
def dummy_shape_spec():
    """Create minimal shape specification."""
    return create_shape_spec(
        shapes={"x": (10,), "y": (5,)},
    )


@pytest.fixture
def dummy_entry_configs():
    """Minimal entry configurations with one model-input feature."""
    return (Feature("x", value=torch.zeros(4, 1)),)


def test_checkpoint_save_includes_metadata(
    patch_factory,
    dummy_wrapper_settings,
    dummy_model_settings,
    dummy_shape_spec,
    dummy_entry_configs,
    tmp_path: Path,
):
    """Test that saved checkpoints include dlkit_metadata."""
    wrapper = StandardLightningWrapper(
        settings=dummy_wrapper_settings,
        model_settings=dummy_model_settings,
        shape_spec=dummy_shape_spec,
        entry_configs=dummy_entry_configs,
    )

    checkpoint = {}
    wrapper.on_save_checkpoint(checkpoint)

    assert "dlkit_metadata" in checkpoint
    assert "wrapper_type" in checkpoint["dlkit_metadata"]


def test_checkpoint_load_rejects_missing_metadata(
    patch_factory,
    dummy_wrapper_settings,
    dummy_model_settings,
    dummy_shape_spec,
    dummy_entry_configs,
):
    """Test that loading a checkpoint without dlkit_metadata raises ValueError."""
    wrapper = StandardLightningWrapper(
        settings=dummy_wrapper_settings,
        model_settings=dummy_model_settings,
        shape_spec=dummy_shape_spec,
        entry_configs=dummy_entry_configs,
    )

    checkpoint = {
        "state_dict": {},
        "hyper_parameters": {},
    }

    with pytest.raises(ValueError, match="Checkpoint missing 'dlkit_metadata'"):
        wrapper.on_load_checkpoint(checkpoint)


def test_checkpoint_load_accepts_legacy_checkpoint(
    patch_factory,
    dummy_wrapper_settings,
    dummy_model_settings,
    dummy_shape_spec,
    dummy_entry_configs,
):
    """Test that older checkpoints are accepted via normalization."""
    wrapper = StandardLightningWrapper(
        settings=dummy_wrapper_settings,
        model_settings=dummy_model_settings,
        shape_spec=dummy_shape_spec,
        entry_configs=dummy_entry_configs,
    )

    checkpoint = {
        "state_dict": {},
        "dlkit_metadata": {
            "model_family": "dlkit_nn",
            "wrapper_type": "StandardLightningWrapper",
            "model_settings": {
                "name": "_DummyModel",
                "module_path": "tests.core.models.wrappers.test_checkpoint_version_validation",
                "params": {"hidden_size": 64},
                "class_name": "ModelComponentSettings",
            },
        },
    }

    # Should not raise — migration handles old format transparently
    wrapper.on_load_checkpoint(checkpoint)


def test_checkpoint_save_is_pure(
    patch_factory,
    dummy_wrapper_settings,
    dummy_model_settings,
    dummy_shape_spec,
    dummy_entry_configs,
):
    """Test that on_save_checkpoint does not mutate wrapper state."""
    wrapper = StandardLightningWrapper(
        settings=dummy_wrapper_settings,
        model_settings=dummy_model_settings,
        shape_spec=dummy_shape_spec,
        entry_configs=dummy_entry_configs,
    )

    initial_entry_configs = wrapper.get_entry_configs()

    checkpoint = {}
    wrapper.on_save_checkpoint(checkpoint)

    assert wrapper.get_entry_configs() == initial_entry_configs
    assert "dlkit_metadata" in checkpoint
