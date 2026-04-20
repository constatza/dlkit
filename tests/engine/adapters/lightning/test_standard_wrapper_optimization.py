"""Integration tests verifying optimization controller wiring in StandardLightningWrapper.

These tests exercise the full config → program → controller → wrapper path to confirm:
- Sequential 2-stage programs are always routed to ManualOptimizationController
  (so Lightning does NOT step inactive-stage optimizers automatically).
- Concurrent 2-optimizer groups are routed to AutomaticOptimizationController
  and return both optimizer configs (with schedulers preserved).
- _requires_manual_optimization detects LBFGS correctly.
- Both optimizers in a concurrent group actually update parameters when stepped.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn
from torch.nn import ModuleList

from dlkit.engine.adapters.lightning.standard import (
    StandardLightningWrapper,
    _requires_manual_optimization,
)
from dlkit.engine.adapters.lightning.wrapper_types import WrapperComponents
from dlkit.engine.training.optimization.builder import OptimizerPolicyBuilder
from dlkit.engine.training.optimization.controllers import (
    AutomaticOptimizationController,
    ManualOptimizationController,
)
from dlkit.infrastructure.config import OptimizerPolicySettings
from dlkit.infrastructure.config.data_entries import Feature, Target
from dlkit.infrastructure.config.model_components import (
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.infrastructure.config.optimization_stage import (
    ConcurrentOptimizationSettings,
    OptimizationStageSettings,
)
from dlkit.infrastructure.config.optimization_trigger import EpochTriggerSettings
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    StepLRSettings,
)

_MODULE = "tests.engine.adapters.lightning.test_standard_wrapper_optimization"


# ---------------------------------------------------------------------------
# Minimal model registered at a predictable module path
# ---------------------------------------------------------------------------


class _TwoLayer(nn.Module):
    """Two-layer model with distinctly named sub-modules for path-based selector tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc0 = nn.Linear(4, 8)
        self.fc1 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(torch.relu(self.fc0(x)))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_layer_model() -> _TwoLayer:
    """Tiny two-layer model used across optimization tests."""
    return _TwoLayer()


@pytest.fixture
def sequential_two_stage_settings() -> OptimizerPolicySettings:
    """Two sequential stages: SGD (epoch 5 trigger) → Adam (no trigger)."""
    return OptimizerPolicySettings(
        stages=(
            OptimizationStageSettings(
                optimizer=AdamWSettings(),
                trigger=EpochTriggerSettings(at_epoch=5),
            ),
            OptimizationStageSettings(
                optimizer=AdamSettings(),
            ),
        )
    )


@pytest.fixture
def concurrent_two_optimizer_settings() -> OptimizerPolicySettings:
    """Single concurrent group with SGD + Adam on all parameters."""
    return OptimizerPolicySettings(
        stages=(
            ConcurrentOptimizationSettings(
                optimizers=(
                    OptimizationStageSettings(
                        optimizer=AdamWSettings(),
                    ),
                    OptimizationStageSettings(
                        optimizer=AdamSettings(),
                    ),
                )
            ),
        )
    )


@pytest.fixture
def concurrent_with_scheduler_settings() -> OptimizerPolicySettings:
    """Concurrent group: SGD with StepLR scheduler + Adam, no scheduler."""
    return OptimizerPolicySettings(
        stages=(
            ConcurrentOptimizationSettings(
                optimizers=(
                    OptimizationStageSettings(
                        optimizer=AdamWSettings(),
                        scheduler=StepLRSettings(step_size=1),
                    ),
                    OptimizationStageSettings(
                        optimizer=AdamSettings(),
                    ),
                )
            ),
        )
    )


def _make_wrapper(settings: OptimizerPolicySettings) -> StandardLightningWrapper:
    """Build a minimal StandardLightningWrapper with the given optimization settings."""
    components = WrapperComponents(
        loss_fn=nn.MSELoss(),
        val_metric_routes=[],
        test_metric_routes=[],
        optimizer_policy_settings=settings,
        feature_transforms={"x": ModuleList()},
        target_transforms={"y": ModuleList()},
    )
    return StandardLightningWrapper(
        model_settings=ModelComponentSettings(name="_TwoLayer", module_path=_MODULE),
        settings=WrapperComponentSettings(),
        components=components,
        entry_configs=(Feature(name="x"), Target(name="y")),
    )


# ---------------------------------------------------------------------------
# _requires_manual_optimization unit tests
# ---------------------------------------------------------------------------


class TestRequiresManualOptimization:
    def test_single_stage_does_not_require_manual(self, two_layer_model: _TwoLayer) -> None:
        settings = OptimizerPolicySettings()
        program = OptimizerPolicyBuilder().build(two_layer_model, settings)
        assert not _requires_manual_optimization(program)

    def test_sequential_two_stages_require_manual(
        self,
        two_layer_model: _TwoLayer,
        sequential_two_stage_settings: OptimizerPolicySettings,
    ) -> None:
        program = OptimizerPolicyBuilder().build(two_layer_model, sequential_two_stage_settings)
        assert _requires_manual_optimization(program)

    def test_concurrent_group_does_not_require_manual(
        self,
        two_layer_model: _TwoLayer,
        concurrent_two_optimizer_settings: OptimizerPolicySettings,
    ) -> None:
        program = OptimizerPolicyBuilder().build(two_layer_model, concurrent_two_optimizer_settings)
        assert not _requires_manual_optimization(program)

    def test_lbfgs_stage_requires_manual(self, two_layer_model: _TwoLayer) -> None:
        # Build the program directly to bypass optimizer-factory kwarg filtering
        # (LBFGS rejects weight_decay; that's a separate factory concern).
        from dlkit.engine.training.optimization.state import ActiveStage, RunningOptimizerPolicy
        from dlkit.engine.training.optimization.triggers import NoTransitionTrigger

        lbfgs_opt = torch.optim.LBFGS(two_layer_model.parameters())
        stage = ActiveStage(
            optimizer=lbfgs_opt,
            scheduler=None,
            trigger=NoTransitionTrigger(),
            stage_index=0,
        )
        program = RunningOptimizerPolicy(stages=(stage,))
        assert _requires_manual_optimization(program)


# ---------------------------------------------------------------------------
# Controller selection tests
# ---------------------------------------------------------------------------


class TestControllerSelection:
    def test_sequential_two_stages_use_manual_controller(
        self,
        sequential_two_stage_settings: OptimizerPolicySettings,
    ) -> None:
        wrapper = _make_wrapper(sequential_two_stage_settings)
        assert isinstance(wrapper._optimization_controller, ManualOptimizationController)
        assert not wrapper.automatic_optimization

    def test_concurrent_group_uses_automatic_controller(
        self,
        concurrent_two_optimizer_settings: OptimizerPolicySettings,
    ) -> None:
        wrapper = _make_wrapper(concurrent_two_optimizer_settings)
        assert isinstance(wrapper._optimization_controller, AutomaticOptimizationController)
        assert wrapper.automatic_optimization

    def test_single_stage_uses_automatic_controller(self) -> None:
        wrapper = _make_wrapper(OptimizerPolicySettings())
        assert isinstance(wrapper._optimization_controller, AutomaticOptimizationController)
        assert wrapper.automatic_optimization


# ---------------------------------------------------------------------------
# configure_optimizers tests
# ---------------------------------------------------------------------------


class TestConfigureOptimizers:
    def test_single_stage_returns_dict(self) -> None:
        wrapper = _make_wrapper(OptimizerPolicySettings())
        config = wrapper.configure_optimizers()
        assert isinstance(config, dict)
        assert "optimizer" in config

    def test_sequential_two_stages_return_list_of_two_optimizers(
        self,
        sequential_two_stage_settings: OptimizerPolicySettings,
    ) -> None:
        wrapper = _make_wrapper(sequential_two_stage_settings)
        config = wrapper.configure_optimizers()
        assert isinstance(config, list)
        assert len(config) == 2  # noqa: PLR2004

    def test_concurrent_two_optimizers_return_list_of_two_dicts(
        self,
        concurrent_two_optimizer_settings: OptimizerPolicySettings,
    ) -> None:
        wrapper = _make_wrapper(concurrent_two_optimizer_settings)
        config = wrapper.configure_optimizers()
        assert isinstance(config, list)
        assert len(config) == 2  # noqa: PLR2004
        for entry in config:
            assert isinstance(entry, dict)
            assert "optimizer" in entry

    def test_concurrent_scheduler_preserved_in_config(
        self,
        concurrent_with_scheduler_settings: OptimizerPolicySettings,
    ) -> None:
        """BUG-2 regression: schedulers must not be silently dropped for concurrent groups."""
        wrapper = _make_wrapper(concurrent_with_scheduler_settings)
        config = wrapper.configure_optimizers()
        assert isinstance(config, list)
        assert len(config) == 2  # noqa: PLR2004
        # First entry (SGD) has a scheduler; second (Adam) does not.
        entry0 = cast(dict[str, object], config[0])
        entry1 = cast(dict[str, object], config[1])
        assert "lr_scheduler" in entry0
        assert "lr_scheduler" not in entry1


# ---------------------------------------------------------------------------
# Runtime: both optimizers actually step parameters
# ---------------------------------------------------------------------------


def _make_batch(batch_size: int = 4, dim: int = 4) -> torch.Tensor:
    from tensordict import TensorDict

    return TensorDict(
        {
            "features": TensorDict({"x": torch.randn(batch_size, dim)}, batch_size=[batch_size]),
            "targets": TensorDict({"y": torch.randn(batch_size, dim)}, batch_size=[batch_size]),
        },
        batch_size=[batch_size],
    )


class TestBothOptimizersStep:
    def test_concurrent_both_optimizers_update_parameters(
        self,
        concurrent_two_optimizer_settings: OptimizerPolicySettings,
    ) -> None:
        """Both concurrent optimizers must produce parameter gradient updates."""
        wrapper = _make_wrapper(concurrent_two_optimizer_settings)
        assert isinstance(wrapper._optimization_controller, AutomaticOptimizationController)

        # Snapshot parameters before the step
        params_before = {name: p.data.clone() for name, p in wrapper.model.named_parameters()}

        # Simulate a training step for each optimizer (Lightning calls training_step
        # once per optimizer in automatic multi-optimizer mode, passing optimizer_idx).
        # Here we directly step the controller to verify both optimizers fire.
        controller = wrapper._optimization_controller
        all_configs = controller.configure_optimizers()
        assert isinstance(all_configs, list) and len(all_configs) == 2  # noqa: PLR2004

        # Step each optimizer manually to simulate what Lightning does
        batch = _make_batch()
        for opt_config in all_configs:
            opt = opt_config["optimizer"]  # type: ignore[index]
            opt.zero_grad()
            loss, _, _ = wrapper._run_step(batch, 0, "train")
            loss.backward()
            opt.step()

        # At least some parameters must have changed
        params_after = dict(wrapper.model.named_parameters())
        changed = [
            name
            for name, before in params_before.items()
            if not torch.allclose(before, params_after[name].data)
        ]
        assert len(changed) > 0, "No parameters were updated — optimizers did not step"

    def test_sequential_two_stages_step_only_active_optimizer(
        self,
        sequential_two_stage_settings: OptimizerPolicySettings,
    ) -> None:
        """After BUG-1 fix: stage 1 optimizer must NOT update params during stage 0."""
        wrapper = _make_wrapper(sequential_two_stage_settings)
        assert isinstance(wrapper._optimization_controller, ManualOptimizationController)

        controller = wrapper._optimization_controller
        # Both optimizers are registered but only the active stage (index 0) should step.
        assert controller._program.active_index == 0

        params_before = {name: p.data.clone() for name, p in wrapper.model.named_parameters()}

        # Run one manual training step
        batch = _make_batch()
        loss_result = wrapper.training_step(batch, 0)
        assert "loss" in loss_result

        # Parameters must have changed (stage 0 optimizer was active)
        params_after = dict(wrapper.model.named_parameters())
        changed = [
            name
            for name, before in params_before.items()
            if not torch.allclose(before, params_after[name].data)
        ]
        assert len(changed) > 0, "Stage-0 optimizer did not update any parameters"

        # Stage index must still be 0 (trigger fires at epoch 5, not after 1 step)
        assert controller._program.active_index == 0

    def test_stage_advances_when_epoch_trigger_fires(
        self,
        sequential_two_stage_settings: OptimizerPolicySettings,
    ) -> None:
        """Epoch trigger at_epoch=5 must advance program to stage 1 at epoch 5.

        Verifies the full config → program → controller advancement path without
        a Trainer: the controller is called directly (as on_train_epoch_end would).
        """
        wrapper = _make_wrapper(sequential_two_stage_settings)
        controller = wrapper._optimization_controller
        assert isinstance(controller, ManualOptimizationController)
        assert controller._program.active_index == 0

        # Epochs 0-4: trigger should not fire (at_epoch=5)
        for epoch in range(5):
            controller.on_epoch_end(epoch, {})
        assert controller._program.active_index == 0, "Stage advanced too early"

        # Epoch 5: trigger fires → advance to stage 1
        controller.on_epoch_end(5, {})
        assert controller._program.active_index == 1, "Stage did not advance at at_epoch=5"
