"""Smoke tests: run Adam+LBFGS (sequential) and Adam+Muon (concurrent) end-to-end.

Models are kept at 2-dimensional inputs/outputs to keep iteration fast.
No Lightning Trainer is used — training_step() is called directly.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from dlkit.engine.adapters.lightning.standard import StandardLightningWrapper
from dlkit.engine.adapters.lightning.wrapper_types import WrapperComponents
from dlkit.engine.training.optimization.builder import OptimizerPolicyBuilder
from dlkit.engine.training.optimization.concurrent_optimizer import ConcurrentOptimizer
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
from dlkit.infrastructure.config.optimization_stage import OptimizationStageSettings
from dlkit.infrastructure.config.optimization_trigger import TriggerSettings
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    ConcurrentOptimizerSettings,
    LBFGSSettings,
    MuonSettings,
)

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


class _CustomRoleModel(nn.Module):
    """Three-layer model with explicit role annotations via IParameterRoleProvider."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(4, 8, bias=False)
        self.hidden = nn.Linear(8, 8, bias=False)
        self.head = nn.Linear(8, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.hidden(self.embed(x))))

    def parameter_roles(self) -> dict[str, object]:
        from dlkit.domain.nn.parameter_roles import ParameterRole

        return {
            "embed.weight": ParameterRole.INPUT,
            "hidden.weight": ParameterRole.HIDDEN,
            "head.weight": ParameterRole.OUTPUT,
        }


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
                    trigger=TriggerSettings(at_epoch=1),
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

        concurrent_optimizer = ConcurrentOptimizer([muon_opt, adam_opt])
        stage = ActiveStage(
            optimizer=concurrent_optimizer,
            scheduler=None,
            trigger=NoTransitionTrigger(),
            stage_index=0,
        )
        return RunningOptimizerPolicy(stages=(stage,))

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

    def test_configure_optimizers_returns_dict_with_concurrent_optimizer(
        self, controller: AutomaticOptimizationController
    ) -> None:
        config = controller.configure_optimizers()
        assert isinstance(config, dict)
        assert "optimizer" in config
        assert isinstance(config["optimizer"], ConcurrentOptimizer)

    def test_both_optimizers_update_parameters(
        self, model: _ThreeLayer, program: RunningOptimizerPolicy
    ) -> None:
        """Step the ConcurrentOptimizer once and confirm fc1.weight and other params changed."""
        params_before = {n: p.data.clone() for n, p in model.named_parameters()}

        x = torch.randn(4, 2)
        y = torch.randn(4, 2)
        loss_fn = nn.MSELoss()

        stage = program.current
        assert isinstance(stage, ActiveStage)
        assert isinstance(stage.optimizer, ConcurrentOptimizer)

        policy = StepAllOptimizers()
        policy.step(stage, lambda: loss_fn(model(x), y))

        params_after = dict(model.named_parameters())
        changed = [
            n for n, b in params_before.items() if not torch.allclose(b, params_after[n].data)
        ]
        assert "fc1.weight" in changed, "Muon did not update fc1.weight"
        assert len(changed) > 1, "Only Muon-eligible weight changed — Adam params untouched"


# ---------------------------------------------------------------------------
# Task A: IParameterRoleProvider escape hatch
# ---------------------------------------------------------------------------


class TestCustomRoleProviderEscapeHatch:
    """Users can implement IParameterRoleProvider to control Muon eligibility for any architecture."""

    @pytest.fixture
    def custom_model(self) -> _CustomRoleModel:
        """Three-layer model that self-annotates its parameter roles via IParameterRoleProvider.

        Returns:
            An nn.Module implementing IParameterRoleProvider with explicit role declarations.
        """
        from dlkit.domain.nn.role_provider import IParameterRoleProvider

        assert issubclass(_CustomRoleModel, IParameterRoleProvider)
        return _CustomRoleModel()

    @pytest.fixture
    def custom_policy(self) -> OptimizerPolicySettings:
        """Concurrent Muon + AdamW policy settings for the custom model.

        Returns:
            An OptimizerPolicySettings with ConcurrentOptimizerSettings as default_optimizer.
        """
        return OptimizerPolicySettings(
            default_optimizer=ConcurrentOptimizerSettings(
                optimizers=(MuonSettings(lr=0.02), AdamWSettings(lr=1e-3))
            )
        )

    def test_custom_role_provider_overrides_ffnn_inference(
        self,
        custom_model: _CustomRoleModel,
        custom_policy: OptimizerPolicySettings,
    ) -> None:
        """IParameterRoleProvider declarations take precedence over FFNN position inference.

        Args:
            custom_model: Model implementing IParameterRoleProvider.
            custom_policy: Config-driven concurrent Muon + AdamW policy.
        """
        program = OptimizerPolicyBuilder().build(custom_model, custom_policy)
        stage = program.current
        assert isinstance(stage, ActiveStage)
        assert isinstance(stage.optimizer, ConcurrentOptimizer)

        sub_opts = stage.optimizer.sub_optimizers
        muon_params = [p for pg in sub_opts[0].param_groups for p in pg["params"]]
        adamw_params = [p for pg in sub_opts[1].param_groups for p in pg["params"]]

        def _in(param: nn.Parameter, param_list: list[nn.Parameter]) -> bool:
            return any(p is param for p in param_list)

        hidden_weight = cast(nn.Parameter, custom_model.hidden.weight)
        embed_weight = cast(nn.Parameter, custom_model.embed.weight)
        head_weight = cast(nn.Parameter, custom_model.head.weight)

        assert _in(hidden_weight, muon_params), "HIDDEN weight must go to Muon"
        assert not _in(embed_weight, muon_params), "INPUT weight must NOT go to Muon"
        assert not _in(head_weight, muon_params), "OUTPUT weight must NOT go to Muon"
        assert _in(embed_weight, adamw_params)
        assert _in(head_weight, adamw_params)


# ---------------------------------------------------------------------------
# Task B: Config-driven Muon constraints
# ---------------------------------------------------------------------------


class TestMuonConstraintsFromConfig:
    """Config-driven Muon build must never route INPUT or OUTPUT layer params to Muon."""

    @pytest.fixture
    def three_layer(self) -> _ThreeLayer:
        """Provide a fresh _ThreeLayer model for each test.

        Returns:
            A _ThreeLayer with fc0 (INPUT), fc1 (HIDDEN), fc2 (OUTPUT).
        """
        return _ThreeLayer()

    @pytest.fixture
    def concurrent_policy(self) -> OptimizerPolicySettings:
        """Concurrent Muon + AdamW policy settings.

        Returns:
            An OptimizerPolicySettings with ConcurrentOptimizerSettings as default_optimizer.
        """
        return OptimizerPolicySettings(
            default_optimizer=ConcurrentOptimizerSettings(
                optimizers=(MuonSettings(lr=0.02), AdamWSettings(lr=1e-3))
            )
        )

    def test_muon_receives_only_hidden_2d_params(
        self,
        three_layer: _ThreeLayer,
        concurrent_policy: OptimizerPolicySettings,
    ) -> None:
        """Only the HIDDEN weight (fc1.weight) reaches Muon; INPUT/OUTPUT weights do not.

        Args:
            three_layer: Three-layer FFNN with fc0/fc1/fc2.
            concurrent_policy: Concurrent Muon + AdamW policy.
        """
        program = OptimizerPolicyBuilder().build(three_layer, concurrent_policy)
        stage = program.current
        assert isinstance(stage, ActiveStage)
        assert isinstance(stage.optimizer, ConcurrentOptimizer)

        sub_opts = stage.optimizer.sub_optimizers
        muon_params = [p for pg in sub_opts[0].param_groups for p in pg["params"]]
        adamw_params = [p for pg in sub_opts[1].param_groups for p in pg["params"]]

        def _in(param: nn.Parameter, param_list: list[nn.Parameter]) -> bool:
            return any(p is param for p in param_list)

        named = dict(three_layer.named_parameters())
        weights_2d = [(n, p) for n, p in named.items() if p.ndim == 2]
        # _ThreeLayer has 3 linear weights: fc0=INPUT, fc1=HIDDEN, fc2=OUTPUT
        _, first_weight = weights_2d[0]
        _, hidden_weight = weights_2d[1]
        _, last_weight = weights_2d[2]

        assert _in(hidden_weight, muon_params), "HIDDEN weight (fc1) must be in Muon"
        assert not _in(first_weight, muon_params), "INPUT weight (fc0) must NOT be in Muon"
        assert not _in(last_weight, muon_params), "OUTPUT weight (fc2) must NOT be in Muon"

        # All biases go to AdamW
        for name, param in named.items():
            if "bias" in name:
                assert _in(param, adamw_params), f"Bias {name} must go to AdamW"

    def test_adamw_receives_input_output_weights_and_biases(
        self,
        three_layer: _ThreeLayer,
        concurrent_policy: OptimizerPolicySettings,
    ) -> None:
        """AdamW receives the INPUT and OUTPUT layer weights and all biases.

        Args:
            three_layer: Three-layer FFNN with fc0/fc1/fc2.
            concurrent_policy: Concurrent Muon + AdamW policy.
        """
        program = OptimizerPolicyBuilder().build(three_layer, concurrent_policy)
        stage = program.current
        assert isinstance(stage, ActiveStage)
        assert isinstance(stage.optimizer, ConcurrentOptimizer)
        adamw_params = [
            p for pg in stage.optimizer.sub_optimizers[1].param_groups for p in pg["params"]
        ]

        def _in(param: nn.Parameter, param_list: list[nn.Parameter]) -> bool:
            return any(p is param for p in param_list)

        named = dict(three_layer.named_parameters())
        weights_2d = [(n, p) for n, p in named.items() if p.ndim == 2]
        _, first_weight = weights_2d[0]
        _, last_weight = weights_2d[2]

        assert _in(first_weight, adamw_params), "INPUT weight (fc0) must be in AdamW"
        assert _in(last_weight, adamw_params), "OUTPUT weight (fc2) must be in AdamW"
