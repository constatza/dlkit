"""Smoke tests: run Adam+LBFGS (sequential) and Adam+Muon (concurrent) end-to-end.

Models are kept at 2-dimensional inputs/outputs to keep iteration fast.
No Lightning Trainer is used — training_step() is called directly.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from dlkit.engine.adapters.lightning.standard import StandardLightningWrapper
from dlkit.engine.adapters.lightning.wrapper_types import WrapperComponents
from dlkit.engine.training.optimization.controllers import (
    AutomaticOptimizationController,
    ManualOptimizationController,
)
from dlkit.engine.training.optimization.inventory import TorchParameterInventory
from dlkit.engine.training.optimization.partitioning import ParameterPartitioner
from dlkit.engine.training.optimization.role_inference import (
    make_default_inference_strategy,
)
from dlkit.engine.training.optimization.selectors import (
    MuonEligibleSelector,
    NonMuonSelector,
)
from dlkit.engine.training.optimization.state import (
    ActiveConcurrentGroup,
    ActiveStage,
    RunningOptimizerPolicy,
)
from dlkit.engine.training.optimization.state_repository import OptimizationStateRepository
from dlkit.engine.training.optimization.stepping import StepAllOptimizers
from dlkit.engine.training.optimization.triggers import NoTransitionTrigger
from dlkit.infrastructure.config import OptimizerPolicySettings
from dlkit.infrastructure.config.data_entries import Feature, Target
from dlkit.infrastructure.config.model_components import (
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.infrastructure.config.optimization_stage import (
    OptimizationStageSettings,
)
from dlkit.infrastructure.config.optimization_trigger import EpochTriggerSettings
from dlkit.infrastructure.config.optimizer_component import AdamSettings, LBFGSSettings

_MODULE = "tests.engine.training.optimization.test_smoke"


# ---------------------------------------------------------------------------
# Minimal models
# ---------------------------------------------------------------------------


class _Linear2D(nn.Module):
    """Single linear layer: 2→2. Used for LBFGS (convex, closed-form)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _ThreeLayer(nn.Module):
    """Three linear layers: 2→4→4→2.

    FFNNRoleInferenceStrategy assigns:
      fc0.weight → INPUT   (2-D, not Muon-eligible)
      fc1.weight → HIDDEN  (2-D, Muon-eligible)
      fc2.weight → OUTPUT  (2-D, not Muon-eligible)
      all biases → BIAS
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc0 = nn.Linear(2, 4)
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(torch.relu(self.fc0(x)))))


# ---------------------------------------------------------------------------
# Shared batch fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def batch_2d() -> TensorDict:
    """Minimal 4-sample batch with 2-dimensional features and targets."""
    return TensorDict(
        {
            "features": TensorDict({"x": torch.randn(4, 2)}, batch_size=[4]),
            "targets": TensorDict({"y": torch.randn(4, 2)}, batch_size=[4]),
        },
        batch_size=[4],
    )


# ---------------------------------------------------------------------------
# Helper: wrap a model in a StandardLightningWrapper
# ---------------------------------------------------------------------------


def _make_wrapper(
    model_cls_name: str,
    opt_settings: OptimizerPolicySettings,
) -> StandardLightningWrapper:
    components = WrapperComponents(
        loss_fn=nn.MSELoss(),
        val_metric_routes=[],
        test_metric_routes=[],
        optimizer_policy_settings=opt_settings,
        feature_transforms={"x": nn.ModuleList()},
        target_transforms={"y": nn.ModuleList()},
    )
    return StandardLightningWrapper(
        model_settings=ModelComponentSettings(name=model_cls_name, module_path=_MODULE),
        settings=WrapperComponentSettings(),
        components=components,
        entry_configs=(Feature(name="x"), Target(name="y")),
    )


# ---------------------------------------------------------------------------
# Smoke test 1: Adam → LBFGS sequential
# ---------------------------------------------------------------------------


class TestAdamThenLBFGS:
    """Two sequential stages: Adam for epoch 0, then LBFGS from epoch 1 onward."""

    @pytest.fixture
    def wrapper(self) -> StandardLightningWrapper:
        settings = OptimizerPolicySettings(
            stages=(
                OptimizationStageSettings(
                    optimizer=AdamSettings(lr=1e-2),
                    trigger=EpochTriggerSettings(at_epoch=1),
                ),
                OptimizationStageSettings(
                    optimizer=LBFGSSettings(),
                ),
            )
        )
        return _make_wrapper("_Linear2D", settings)

    def test_uses_manual_controller(self, wrapper: StandardLightningWrapper) -> None:
        assert isinstance(wrapper._optimization_controller, ManualOptimizationController)
        assert not wrapper.automatic_optimization

    def test_stage_0_is_adam(self, wrapper: StandardLightningWrapper) -> None:
        ctrl = wrapper._optimization_controller
        assert isinstance(ctrl, ManualOptimizationController)
        assert ctrl._program.active_index == 0
        assert "adam" in ctrl._program.current.optimizer.__class__.__name__.lower()

    def test_adam_step_updates_parameters(
        self, wrapper: StandardLightningWrapper, batch_2d: TensorDict
    ) -> None:
        params_before = {n: p.data.clone() for n, p in wrapper.model.named_parameters()}
        wrapper.training_step(batch_2d, 0)
        params_after = dict(wrapper.model.named_parameters())
        changed = [
            n for n, b in params_before.items() if not torch.allclose(b, params_after[n].data)
        ]
        assert changed, "Adam did not update any parameters"

    def test_stage_advances_to_lbfgs_at_epoch_1(
        self, wrapper: StandardLightningWrapper, batch_2d: TensorDict
    ) -> None:
        ctrl = wrapper._optimization_controller
        assert isinstance(ctrl, ManualOptimizationController)

        # Simulate one training step followed by epoch-end stage advancement.
        # Call on_train_epoch_end via the controller directly to avoid needing
        # an attached Lightning Trainer (which sets callback_metrics).
        wrapper.training_step(batch_2d, 0)
        ctrl.on_epoch_end(0, {})  # epoch 0 ends; EpochTrigger fires at at_epoch=1
        ctrl.on_epoch_end(1, {})  # epoch 1 ends; trigger fires → advance to LBFGS

        assert ctrl._program.active_index == 1
        assert "lbfgs" in ctrl._program.current.optimizer.__class__.__name__.lower()

    def test_lbfgs_step_updates_parameters(
        self, wrapper: StandardLightningWrapper, batch_2d: TensorDict
    ) -> None:
        ctrl = wrapper._optimization_controller
        assert isinstance(ctrl, ManualOptimizationController)

        # Advance to LBFGS stage
        ctrl.on_epoch_end(1, {})
        assert ctrl._program.active_index == 1

        params_before = {n: p.data.clone() for n, p in wrapper.model.named_parameters()}
        wrapper.training_step(batch_2d, 0)
        params_after = dict(wrapper.model.named_parameters())
        changed = [
            n for n, b in params_before.items() if not torch.allclose(b, params_after[n].data)
        ]
        assert changed, "LBFGS did not update any parameters"


# ---------------------------------------------------------------------------
# Smoke test 2: Adam + Muon concurrent
# ---------------------------------------------------------------------------


class TestAdamAndMuonConcurrent:
    """Concurrent group: Muon on HIDDEN weights, Adam on everything else.

    The program is built directly (not via config) because the builder's
    config path doesn't yet resolve ParameterSelectorSettings to
    IParameterSelector instances. Selectors are injected as real objects.
    """

    @pytest.fixture
    def model(self) -> _ThreeLayer:
        return _ThreeLayer()

    @pytest.fixture
    def program(self, model: _ThreeLayer) -> RunningOptimizerPolicy:
        """Build a concurrent program with Muon on fc1.weight, Adam on rest."""
        strategy = make_default_inference_strategy(model)
        inventory = TorchParameterInventory(
            model,
            role_resolver=lambda d: strategy.infer(model, d.name, d.parameter) or d.role,
        )

        muon_selector = MuonEligibleSelector()
        adam_selector = NonMuonSelector()

        partitioner = ParameterPartitioner()
        muon_params, adam_params = partitioner.partition(inventory, [muon_selector, adam_selector])

        muon_opt = torch.optim.Muon([{"params": [d.parameter for d in muon_params]}])
        adam_opt = torch.optim.Adam([{"params": [d.parameter for d in adam_params]}], lr=1e-3)

        muon_stage = ActiveStage(
            optimizer=muon_opt, scheduler=None, trigger=NoTransitionTrigger(), stage_index=0
        )
        adam_stage = ActiveStage(
            optimizer=adam_opt, scheduler=None, trigger=NoTransitionTrigger(), stage_index=1
        )
        group = ActiveConcurrentGroup(
            stages=(muon_stage, adam_stage), trigger=NoTransitionTrigger(), group_index=0
        )
        return RunningOptimizerPolicy(stages=(group,))

    @pytest.fixture
    def controller(self, program: RunningOptimizerPolicy) -> AutomaticOptimizationController:
        return AutomaticOptimizationController(program, OptimizationStateRepository())

    def test_muon_eligible_params_identified(self, model: _ThreeLayer) -> None:
        """fc1.weight must be the only Muon-eligible parameter."""
        strategy = make_default_inference_strategy(model)
        inventory = TorchParameterInventory(
            model,
            role_resolver=lambda d: strategy.infer(model, d.name, d.parameter) or d.role,
        )
        selector = MuonEligibleSelector()
        eligible = [d for d in inventory.list_parameters() if selector.is_satisfied_by(d)]
        assert len(eligible) == 1
        assert eligible[0].name == "fc1.weight"

    def test_configure_optimizers_returns_two_dicts(
        self, controller: AutomaticOptimizationController
    ) -> None:
        config = controller.configure_optimizers()
        assert isinstance(config, list)
        assert len(config) == 2  # noqa: PLR2004
        for entry in config:
            assert isinstance(entry, dict) and "optimizer" in entry

    def test_both_optimizers_update_parameters(
        self, model: _ThreeLayer, program: RunningOptimizerPolicy
    ) -> None:
        """Step both optimizers once and confirm fc1.weight and other params changed."""
        params_before = {n: p.data.clone() for n, p in model.named_parameters()}

        x = torch.randn(4, 2)
        y = torch.randn(4, 2)
        loss_fn = nn.MSELoss()

        group = program.current
        assert isinstance(group, ActiveConcurrentGroup)

        policy = StepAllOptimizers()
        policy.step(group, lambda: loss_fn(model(x), y))

        params_after = dict(model.named_parameters())
        changed = [
            n for n, b in params_before.items() if not torch.allclose(b, params_after[n].data)
        ]
        assert "fc1.weight" in changed, "Muon did not update fc1.weight"
        assert len(changed) > 1, "Only Muon-eligible weight changed — Adam params untouched"
