"""Integration test: entry-shapes model checkpoint round-trip."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import ModuleList

from dlkit.common.errors import WorkflowError
from dlkit.common.shapes import ShapeContext
from dlkit.engine.adapters.lightning.concerns._checkpoint_serializer_helpers import (
    deserialize_shapes,
)
from dlkit.engine.adapters.lightning.standard import StandardLightningWrapper
from dlkit.engine.adapters.lightning.wrapper_types import WrapperComponents
from dlkit.engine.inference.model_builder import build_model_from_checkpoint
from dlkit.infrastructure.config import (
    ModelComponentSettings,
    OptimizerPolicySettings,
    WrapperComponentSettings,
)
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.entry_types import ValueEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_IN_SHAPE: tuple[int, ...] = (4,)
_OUT_SHAPE: tuple[int, ...] = (2,)
_BATCH_SIZE: int = 1


@pytest.fixture
def input_shapes() -> dict[str, tuple[int, ...]]:
    """Feature-name-to-shape mapping used to build the model."""
    return {"x": _IN_SHAPE}


@pytest.fixture
def output_shapes() -> dict[str, tuple[int, ...]]:
    """Target-name-to-shape mapping used to build the model."""
    return {"y": _OUT_SHAPE}


@pytest.fixture
def model_settings() -> ModelComponentSettings:
    """ModelComponentSettings pointing at FactorizedLinearNetwork."""
    return ModelComponentSettings(
        name="FactorizedLinearNetwork",
        module_path="dlkit.domain.nn.ffnn.linear",
    )


@pytest.fixture
def wrapper_settings() -> WrapperComponentSettings:
    """Minimal wrapper settings."""
    return WrapperComponentSettings()


@pytest.fixture
def entry_configs():
    """Single feature entry for the integration test."""
    return (
        ValueEntry(
            name="x", value=torch.zeros(_BATCH_SIZE, _IN_SHAPE[0]), data_role=DataRole.FEATURE
        ),
    )


@pytest.fixture
def components(entry_configs) -> WrapperComponents:
    """Pre-built WrapperComponents for StandardLightningWrapper."""
    return WrapperComponents(
        loss_fn=nn.MSELoss(),
        val_metric_routes=[],
        test_metric_routes=[],
        optimizer_policy_settings=OptimizerPolicySettings(),
        feature_transforms={e.name: ModuleList() for e in entry_configs if e.name},
        target_transforms={},
    )


@pytest.fixture
def flat_model_settings() -> ModelComponentSettings:
    """ModelComponentSettings with canonical flat hyperparameters (no nested params)."""
    return ModelComponentSettings.model_validate(
        {
            "class": "Linear",
            "module_path": "torch.nn",
            "in_features": 4,
            "out_features": 2,
        }
    )


@pytest.fixture
def flat_entry_configs():
    """Feature/target entries for the flat-settings round-trip test."""
    return (
        ValueEntry(name="input", value=torch.zeros(_BATCH_SIZE, 4), data_role=DataRole.FEATURE),
        ValueEntry(name="y", value=torch.zeros(_BATCH_SIZE, 2), data_role=DataRole.TARGET),
    )


@pytest.fixture
def flat_components(flat_entry_configs) -> WrapperComponents:
    """Pre-built WrapperComponents for the flat-settings round-trip test."""
    return WrapperComponents(
        loss_fn=nn.MSELoss(),
        val_metric_routes=[],
        test_metric_routes=[],
        optimizer_policy_settings=OptimizerPolicySettings(),
        feature_transforms={e.name: ModuleList() for e in flat_entry_configs if e.name == "input"},
        target_transforms={},
    )


@pytest.fixture
def malformed_nested_params_checkpoint() -> dict:
    """Obsolete checkpoint payload with hyper_kwargs.params nesting."""
    return {
        "dlkit_metadata": {
            "model_settings": {
                "name": "Linear",
                "module_path": "torch.nn",
                "hyper_kwargs": {"params": {"in_features": 4, "out_features": 2}},
            }
        }
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_shapes_checkpoint_round_trip(
    input_shapes, output_shapes, model_settings, wrapper_settings, entry_configs, components
):
    """Full save→reconstruct cycle: entry shapes persist to checkpoint and reload."""
    wrapper = StandardLightningWrapper(
        settings=wrapper_settings,
        model_settings=model_settings,
        components=components,
        entry_configs=entry_configs,
        context=ShapeContext(input_shapes, output_shapes),
    )

    # Build a Lightning-style checkpoint with prefixed state dict keys
    state_dict = {"model." + k: v for k, v in wrapper.model.state_dict().items()}
    checkpoint: dict = {"state_dict": state_dict}
    wrapper.on_save_checkpoint(checkpoint)

    # Assert shapes were persisted under the new metadata keys
    assert checkpoint["dlkit_metadata"]["input_shapes"] == {"x": list(_IN_SHAPE)}
    assert checkpoint["dlkit_metadata"]["output_shapes"] == {"y": list(_OUT_SHAPE)}

    # Reconstruct shape mappings from checkpoint and rebuild model
    restored_inputs = deserialize_shapes(checkpoint["dlkit_metadata"]["input_shapes"])
    restored_outputs = deserialize_shapes(checkpoint["dlkit_metadata"]["output_shapes"])
    assert restored_inputs == {"x": _IN_SHAPE}
    assert restored_outputs == {"y": _OUT_SHAPE}

    model = build_model_from_checkpoint(
        checkpoint, input_shapes=restored_inputs, output_shapes=restored_outputs
    )

    # Verify model produces correct output shape
    x = torch.zeros(_BATCH_SIZE, _IN_SHAPE[0])
    output = model(x)
    assert output.shape == torch.Size([_BATCH_SIZE, _OUT_SHAPE[0]])


def test_flat_model_settings_checkpoint_round_trip_reconstructs_model(
    flat_model_settings, flat_entry_configs, flat_components
):
    """Canonical flat model kwargs should survive checkpoint round-trip."""
    wrapper = StandardLightningWrapper(
        settings=WrapperComponentSettings(),
        model_settings=flat_model_settings,
        components=flat_components,
        entry_configs=flat_entry_configs,
    )

    checkpoint = {"state_dict": {"model." + k: v for k, v in wrapper.model.state_dict().items()}}
    wrapper.on_save_checkpoint(checkpoint)

    rebuilt = build_model_from_checkpoint(checkpoint, None, None)

    assert isinstance(rebuilt, nn.Linear)
    assert rebuilt.in_features == 4
    assert rebuilt.out_features == 2


def test_malformed_nested_params_checkpoint_fails_fast(malformed_nested_params_checkpoint):
    """Obsolete checkpoint payloads with nested params must fail with a targeted error."""
    with pytest.raises(WorkflowError, match="unsupported nested 'params'"):
        build_model_from_checkpoint(malformed_nested_params_checkpoint, None, None)
