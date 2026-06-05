"""Integration test: contract-based model checkpoint round-trip."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import ModuleList

from dlkit.domain.nn.contracts import TabulaRSpec, deserialize_contract
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
def contract() -> TabulaRSpec:
    """TabulaRSpec contract used for building the model."""
    return TabulaRSpec(in_shape=_IN_SHAPE, out_shape=_OUT_SHAPE)


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


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_contract_checkpoint_round_trip(
    contract, model_settings, wrapper_settings, entry_configs, components
):
    """Full save→reconstruct cycle: contract persists to checkpoint and reloads."""
    wrapper = StandardLightningWrapper(
        settings=wrapper_settings,
        model_settings=model_settings,
        components=components,
        entry_configs=entry_configs,
        contract=contract,
    )

    # Build a Lightning-style checkpoint with prefixed state dict keys
    state_dict = {"model." + k: v for k, v in wrapper.model.state_dict().items()}
    checkpoint: dict = {"state_dict": state_dict}
    wrapper.on_save_checkpoint(checkpoint)

    # Assert contract was persisted
    assert checkpoint["dlkit_metadata"]["contract"]["_type"] == "TabulaRSpec"

    # Reconstruct contract from checkpoint and rebuild model
    restored_contract = deserialize_contract(checkpoint["dlkit_metadata"]["contract"])
    assert restored_contract is not None

    model = build_model_from_checkpoint(checkpoint, contract=restored_contract)

    # Verify model produces correct output shape
    x = torch.zeros(_BATCH_SIZE, _IN_SHAPE[0])
    output = model(x)
    assert output.shape == torch.Size([_BATCH_SIZE, _OUT_SHAPE[0]])
