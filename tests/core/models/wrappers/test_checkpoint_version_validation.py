"""Test checkpoint version validation in ProcessingLightningWrapper."""

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
        # For loss/metrics/optimizer/scheduler
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


def test_checkpoint_save_includes_version(
    patch_factory,
    dummy_wrapper_settings,
    dummy_model_settings,
    dummy_shape_spec,
    tmp_path: Path,
):
    """Test that saved checkpoints include version field."""
    wrapper = StandardLightningWrapper(
        settings=dummy_wrapper_settings,
        model_settings=dummy_model_settings,
        shape_spec=dummy_shape_spec,
    )

    # Create a checkpoint dict
    checkpoint = {}
    wrapper.on_save_checkpoint(checkpoint)

    # Verify version field exists
    assert "dlkit_metadata" in checkpoint
    assert "version" in checkpoint["dlkit_metadata"]
    assert checkpoint["dlkit_metadata"]["version"] == "2.0"


def test_checkpoint_load_rejects_missing_metadata(
    patch_factory,
    dummy_wrapper_settings,
    dummy_model_settings,
    dummy_shape_spec,
):
    """Test that loading a checkpoint without dlkit_metadata raises ValueError."""
    wrapper = StandardLightningWrapper(
        settings=dummy_wrapper_settings,
        model_settings=dummy_model_settings,
        shape_spec=dummy_shape_spec,
    )

    # Create a legacy checkpoint without dlkit_metadata
    checkpoint = {
        "state_dict": {},
        "hyper_parameters": {},
    }

    with pytest.raises(ValueError, match="Checkpoint missing 'dlkit_metadata'"):
        wrapper.on_load_checkpoint(checkpoint)


def test_checkpoint_load_rejects_missing_version(
    patch_factory,
    dummy_wrapper_settings,
    dummy_model_settings,
    dummy_shape_spec,
):
    """Test that loading a checkpoint without version field raises ValueError."""
    wrapper = StandardLightningWrapper(
        settings=dummy_wrapper_settings,
        model_settings=dummy_model_settings,
        shape_spec=dummy_shape_spec,
    )

    # Create a checkpoint with dlkit_metadata but no version
    checkpoint = {
        "state_dict": {},
        "dlkit_metadata": {
            "model_family": "dlkit_nn",
            "wrapper_type": "StandardLightningWrapper",
        },
    }

    with pytest.raises(ValueError, match="missing 'version' field"):
        wrapper.on_load_checkpoint(checkpoint)


def test_checkpoint_load_rejects_unsupported_version(
    patch_factory,
    dummy_wrapper_settings,
    dummy_model_settings,
    dummy_shape_spec,
):
    """Test that loading a checkpoint with unsupported version raises ValueError."""
    wrapper = StandardLightningWrapper(
        settings=dummy_wrapper_settings,
        model_settings=dummy_model_settings,
        shape_spec=dummy_shape_spec,
    )

    # Create a checkpoint with an old version
    checkpoint = {
        "state_dict": {},
        "dlkit_metadata": {
            "version": "1.0",
            "model_family": "dlkit_nn",
            "wrapper_type": "StandardLightningWrapper",
        },
    }

    with pytest.raises(ValueError, match="Unsupported checkpoint version '1.0'"):
        wrapper.on_load_checkpoint(checkpoint)


def test_checkpoint_load_accepts_supported_version(
    patch_factory,
    dummy_wrapper_settings,
    dummy_model_settings,
    dummy_shape_spec,
):
    """Test that loading a checkpoint with version 2.0 succeeds."""
    wrapper = StandardLightningWrapper(
        settings=dummy_wrapper_settings,
        model_settings=dummy_model_settings,
        shape_spec=dummy_shape_spec,
    )

    # Create a valid v2.0 checkpoint
    checkpoint = {
        "state_dict": {},
        "dlkit_metadata": {
            "version": "2.0",
            "model_family": "dlkit_nn",
            "wrapper_type": "StandardLightningWrapper",
        },
    }

    # Should not raise
    wrapper.on_load_checkpoint(checkpoint)


def test_checkpoint_save_is_pure(
    patch_factory,
    dummy_wrapper_settings,
    dummy_model_settings,
    dummy_shape_spec,
):
    """Test that on_save_checkpoint does not mutate wrapper state."""
    wrapper = StandardLightningWrapper(
        settings=dummy_wrapper_settings,
        model_settings=dummy_model_settings,
        shape_spec=dummy_shape_spec,
    )

    # Capture initial shape_spec reference
    initial_shape_spec = wrapper.shape_spec

    # Save checkpoint
    checkpoint = {}
    wrapper.on_save_checkpoint(checkpoint)

    # Verify shape_spec reference hasn't changed (no mutation)
    assert wrapper.shape_spec is initial_shape_spec

    # Verify checkpoint was populated
    assert "dlkit_metadata" in checkpoint
