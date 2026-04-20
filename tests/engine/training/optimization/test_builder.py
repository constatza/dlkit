"""Tests for the OptimizerPolicyBuilder."""

from __future__ import annotations

import torch
import torch.nn as nn

from dlkit.engine.training.optimization.builder import OptimizerPolicyBuilder
from dlkit.engine.training.optimization.state import ActiveConcurrentGroup, ActiveStage
from dlkit.engine.training.optimization.triggers import (
    EpochTransitionTrigger,
    PlateauTransitionTrigger,
)
from dlkit.infrastructure.config.optimization_selector import (
    ModulePathSelectorSettings,
    RoleSelectorSettings,
)
from dlkit.infrastructure.config.optimization_stage import (
    ConcurrentOptimizationSettings,
    OptimizationStageSettings,
)
from dlkit.infrastructure.config.optimization_trigger import (
    EpochTriggerSettings,
    PlateauTriggerSettings,
)
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    StepLRSettings,
)
from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings


class TestOptimizerPolicyBuilder:
    """Tests for OptimizerPolicyBuilder."""

    def test_builder_degenerate_single_stage(
        self,
        tiny_model: nn.Sequential,
    ) -> None:
        """Verify builder creates single stage from empty settings.

        Args:
            tiny_model: Tiny model fixture.
        """
        # Empty stages tuple -> defaults to AdamW optimizer
        settings = OptimizerPolicySettings()
        builder = OptimizerPolicyBuilder()

        program = builder.build(tiny_model, settings)

        assert len(program.stages) == 1
        assert isinstance(program.stages[0], ActiveStage)
        assert program.stages[0].name == "default"

    def test_builder_explicit_single_stage(
        self,
        tiny_model: nn.Sequential,
    ) -> None:
        """Verify builder creates single explicit stage.

        Args:
            tiny_model: Tiny model fixture.
        """
        stage_config = OptimizationStageSettings(
            optimizer=AdamWSettings(),
        )
        settings = OptimizerPolicySettings(stages=(stage_config,))
        builder = OptimizerPolicyBuilder()

        program = builder.build(tiny_model, settings)

        assert len(program.stages) == 1
        assert isinstance(program.stages[0], ActiveStage)

    def test_builder_two_sequential_stages(
        self,
        tiny_model: nn.Sequential,
    ) -> None:
        """Verify builder creates two sequential stages with correct indices.

        Args:
            tiny_model: Tiny model fixture.
        """
        stage_0_config = OptimizationStageSettings(
            optimizer=AdamWSettings(),
        )
        stage_1_config = OptimizationStageSettings(
            optimizer=AdamSettings(),
        )
        settings = OptimizerPolicySettings(
            stages=(stage_0_config, stage_1_config),
        )
        builder = OptimizerPolicyBuilder()

        program = builder.build(tiny_model, settings)

        assert len(program.stages) == 2
        stage_0 = program.stages[0]
        stage_1 = program.stages[1]
        assert isinstance(stage_0, ActiveStage)
        assert isinstance(stage_1, ActiveStage)
        assert stage_0.stage_index == 0
        assert stage_1.stage_index == 1

    def test_builder_epoch_trigger_stage(
        self,
        tiny_model: nn.Sequential,
    ) -> None:
        """Verify builder creates epoch trigger correctly.

        Args:
            tiny_model: Tiny model fixture.
        """
        trigger_config = EpochTriggerSettings(at_epoch=10)
        stage_config = OptimizationStageSettings(
            optimizer=AdamWSettings(),
            trigger=trigger_config,
        )
        settings = OptimizerPolicySettings(stages=(stage_config,))
        builder = OptimizerPolicyBuilder()

        program = builder.build(tiny_model, settings)

        stage = program.stages[0]
        assert isinstance(stage, ActiveStage)
        assert isinstance(stage.trigger, EpochTransitionTrigger)
        # Verify trigger's at_epoch attribute
        assert stage.trigger._at_epoch == 10

    def test_builder_default_optimizer_is_adamw(
        self,
        tiny_model: nn.Sequential,
    ) -> None:
        """Verify builder uses AdamW as default optimizer.

        Args:
            tiny_model: Tiny model fixture.
        """
        settings = OptimizerPolicySettings()
        builder = OptimizerPolicyBuilder()

        program = builder.build(tiny_model, settings)

        stage_entry = program.stages[0]
        assert isinstance(stage_entry, ActiveStage)
        # Verify it's an Adam optimizer (AdamW is a variant of Adam)
        assert isinstance(stage_entry.optimizer, torch.optim.AdamW)

    def test_builder_concurrent_group(self, tiny_model: nn.Sequential) -> None:
        """Verify builder produces ActiveConcurrentGroup with correct inner stage indices.

        Args:
            tiny_model: Tiny two-layer model fixture.
        """
        settings = OptimizerPolicySettings(
            stages=(
                ConcurrentOptimizationSettings(
                    optimizers=(
                        OptimizationStageSettings(optimizer=AdamWSettings()),
                        OptimizationStageSettings(optimizer=AdamSettings()),
                    )
                ),
            )
        )
        program = OptimizerPolicyBuilder().build(tiny_model, settings)

        assert len(program.stages) == 1
        group = program.stages[0]
        assert isinstance(group, ActiveConcurrentGroup)
        assert len(group.stages) == 2
        assert group.stages[0].stage_index == 0
        assert group.stages[1].stage_index == 1
        assert isinstance(group.stages[0].optimizer, torch.optim.AdamW)
        assert isinstance(group.stages[1].optimizer, torch.optim.Adam)

    def test_builder_plateau_trigger_stage(self, tiny_model: nn.Sequential) -> None:
        """Verify builder creates PlateauTransitionTrigger with correct parameters.

        Args:
            tiny_model: Tiny two-layer model fixture.
        """
        settings = OptimizerPolicySettings(
            stages=(
                OptimizationStageSettings(
                    optimizer=AdamWSettings(),
                    trigger=PlateauTriggerSettings(monitor="val_loss", patience=5, min_delta=1e-3),
                ),
                OptimizationStageSettings(optimizer=AdamSettings()),
            )
        )
        program = OptimizerPolicyBuilder().build(tiny_model, settings)

        stage = program.stages[0]
        assert isinstance(stage, ActiveStage)
        assert isinstance(stage.trigger, PlateauTransitionTrigger)
        assert stage.trigger._patience == 5
        assert stage.trigger._monitor == "val_loss"
        assert stage.trigger._min_delta == 1e-3

    def test_builder_stage_with_scheduler(self, tiny_model: nn.Sequential) -> None:
        """Verify builder attaches scheduler to stage and wires monitor/frequency.

        Args:
            tiny_model: Tiny two-layer model fixture.
        """
        settings = OptimizerPolicySettings(
            stages=(
                OptimizationStageSettings(
                    optimizer=AdamWSettings(),
                    scheduler=StepLRSettings(step_size=5, gamma=0.5),
                ),
            )
        )
        program = OptimizerPolicyBuilder().build(tiny_model, settings)

        stage = program.stages[0]
        assert isinstance(stage, ActiveStage)
        assert stage.scheduler is not None
        assert isinstance(stage.scheduler, torch.optim.lr_scheduler.StepLR)

    def test_builder_module_path_selector(self, tiny_model: nn.Sequential) -> None:
        """Stage with ModulePathSelectorSettings selects only the matching sub-module params.

        tiny_model is Sequential(Linear(4,8), Linear(8,2)).
        module_path for "0.*" params is "0"; for "1.*" params is "1".

        Args:
            tiny_model: Tiny two-layer model fixture.
        """
        settings = OptimizerPolicySettings(
            stages=(
                OptimizationStageSettings(
                    optimizer=AdamWSettings(),
                    selector=ModulePathSelectorSettings(prefix="0"),
                ),
            )
        )
        program = OptimizerPolicyBuilder().build(tiny_model, settings)

        stage = program.stages[0]
        assert isinstance(stage, ActiveStage)
        opt_params = [p for group in stage.optimizer.param_groups for p in group["params"]]
        # Only the first linear layer: weight (4x8=32) + bias (8) = 2 tensors
        assert len(opt_params) == 2  # noqa: PLR2004
        # The stage-1 params must NOT be in the optimizer
        layer1_weight = dict(tiny_model.named_parameters())["1.weight"]
        assert all(p is not layer1_weight for p in opt_params)

    def test_builder_role_selector_unknown_selects_all(self, tiny_model: nn.Sequential) -> None:
        """RoleSelectorSettings(role='unknown') selects all params when no role resolver is used.

        The builder's TorchParameterInventory assigns UNKNOWN to all params by default.

        Args:
            tiny_model: Tiny two-layer model fixture.
        """
        settings = OptimizerPolicySettings(
            stages=(
                OptimizationStageSettings(
                    optimizer=AdamWSettings(),
                    selector=RoleSelectorSettings(role="unknown"),
                ),
            )
        )
        program = OptimizerPolicyBuilder().build(tiny_model, settings)

        stage = program.stages[0]
        assert isinstance(stage, ActiveStage)
        opt_params = [p for group in stage.optimizer.param_groups for p in group["params"]]
        total_params = list(tiny_model.parameters())
        assert len(opt_params) == len(total_params)
