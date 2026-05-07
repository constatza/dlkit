"""Tests for the OptimizerPolicyBuilder."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dlkit.common.errors import ParameterPartitionError
from dlkit.engine.training.optimization.builder import OptimizerPolicyBuilder
from dlkit.engine.training.optimization.concurrent_optimizer import ConcurrentOptimizer
from dlkit.engine.training.optimization.state import ActiveStage
from dlkit.engine.training.optimization.triggers import (
    EpochTransitionTrigger,
    PlateauTransitionTrigger,
)
from dlkit.infrastructure.config.optimization_selector import ParameterSelectorSettings
from dlkit.infrastructure.config.optimization_stage import OptimizationStageSettings
from dlkit.infrastructure.config.optimization_trigger import TriggerSettings
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    BatchedMuonSettings,
    ConcurrentOptimizerSettings,
    MuonSettings,
    ReduceLROnPlateauSettings,
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
        trigger_config = TriggerSettings(at_epoch=10)
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
        """Verify builder produces an ActiveStage with ConcurrentOptimizer for concurrent config.

        tiny_model is Sequential(Linear(4,8), Linear(8,2)) — submodule "0" and "1".
        Explicit selectors partition params to avoid duplicate-param PyTorch warnings.

        Args:
            tiny_model: Tiny two-layer model fixture.
        """
        settings = OptimizerPolicySettings(
            default_optimizer=ConcurrentOptimizerSettings(
                optimizers=(AdamWSettings(), AdamSettings()),
                selectors=(
                    ParameterSelectorSettings(prefix="0"),
                    ParameterSelectorSettings(prefix="1"),
                ),
            )
        )
        program = OptimizerPolicyBuilder().build(tiny_model, settings)

        assert len(program.stages) == 1
        stage = program.stages[0]
        assert isinstance(stage, ActiveStage)
        assert isinstance(stage.optimizer, ConcurrentOptimizer)
        sub_opts = stage.optimizer.sub_optimizers
        assert len(sub_opts) == 2  # noqa: PLR2004
        assert isinstance(sub_opts[0], torch.optim.AdamW)
        assert isinstance(sub_opts[1], torch.optim.Adam)

    def test_builder_batched_muon_default_optimizer(self, tiny_model: nn.Sequential) -> None:
        """Verify lone BatchedMuon default optimizer becomes a concurrent Muon-family split."""
        settings = OptimizerPolicySettings(default_optimizer=BatchedMuonSettings(lr=0.02))
        program = OptimizerPolicyBuilder().build(tiny_model, settings)

        assert len(program.stages) == 1
        stage = program.stages[0]
        assert isinstance(stage, ActiveStage)
        assert isinstance(stage.optimizer, ConcurrentOptimizer)
        assert len(stage.optimizer.sub_optimizers) == 2  # noqa: PLR2004

    def test_builder_muon_stage_auto_splits_params(self, tiny_model: nn.Sequential) -> None:
        """Lone MuonSettings in an explicit stage auto-splits like the default path.

        tiny_model has biases (1-D params) that torch.optim.Muon rejects.
        The builder must apply the same MuonEligibleSelector / NonMuonSelector
        split it uses in _build_default, producing a ConcurrentOptimizer.

        Args:
            tiny_model: Tiny two-layer model fixture.
        """
        settings = OptimizerPolicySettings(
            stages=(OptimizationStageSettings(optimizer=MuonSettings()),)
        )
        program = OptimizerPolicyBuilder().build(tiny_model, settings)

        assert len(program.stages) == 1
        stage = program.stages[0]
        assert isinstance(stage, ActiveStage)
        assert isinstance(stage.optimizer, ConcurrentOptimizer)

    def test_builder_module_path_selector_with_concurrent_covers_all_params(
        self, tiny_model: nn.Sequential
    ) -> None:
        """ConcurrentOptimizerSettings with complementary prefix selectors routes params correctly.

        The correct pattern for module-path selective optimization: two sub-optimizers whose
        prefix selectors are complementary so every parameter is covered exactly once.
        tiny_model is Sequential(Linear(4,8), Linear(8,2)) — submodule "0" and "1".

        Args:
            tiny_model: Tiny two-layer model fixture.
        """
        settings = OptimizerPolicySettings(
            default_optimizer=ConcurrentOptimizerSettings(
                optimizers=(AdamWSettings(), AdamWSettings()),
                selectors=(
                    ParameterSelectorSettings(prefix="0"),
                    ParameterSelectorSettings(prefix="1"),
                ),
            )
        )
        program = OptimizerPolicyBuilder().build(tiny_model, settings)

        stage = program.stages[0]
        assert isinstance(stage.optimizer, ConcurrentOptimizer)
        sub_opts = stage.optimizer.sub_optimizers
        assert len(sub_opts) == 2  # noqa: PLR2004

        named = dict(tiny_model.named_parameters())
        layer0_ids = {id(named["0.weight"]), id(named["0.bias"])}
        layer1_ids = {id(named["1.weight"]), id(named["1.bias"])}

        sub0_ids = {id(p) for pg in sub_opts[0].param_groups for p in pg["params"]}
        sub1_ids = {id(p) for pg in sub_opts[1].param_groups for p in pg["params"]}

        assert sub0_ids == layer0_ids, "sub-optimizer 0 must contain exactly layer-0 params"
        assert sub1_ids == layer1_ids, "sub-optimizer 1 must contain exactly layer-1 params"

    def test_builder_plateau_trigger_stage(self, tiny_model: nn.Sequential) -> None:
        """Verify builder creates PlateauTransitionTrigger with correct parameters.

        Args:
            tiny_model: Tiny two-layer model fixture.
        """
        settings = OptimizerPolicySettings(
            stages=(
                OptimizationStageSettings(
                    optimizer=AdamWSettings(),
                    trigger=TriggerSettings(patience=5, monitor="val_loss", min_delta=1e-3),
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

    def test_builder_default_scheduler_exists_at_runtime(self, tiny_model: nn.Sequential) -> None:
        """Verify default_scheduler creates a live scheduler on the default stage."""
        settings = OptimizerPolicySettings(
            default_optimizer=AdamWSettings(),
            default_scheduler=ReduceLROnPlateauSettings(patience=2, factor=0.5),
        )
        program = OptimizerPolicyBuilder().build(tiny_model, settings)

        stage = program.stages[0]
        assert isinstance(stage, ActiveStage)
        assert stage.scheduler is not None
        assert isinstance(stage.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_builder_all_configured_stage_schedulers_exist_at_runtime(
        self, tiny_model: nn.Sequential
    ) -> None:
        """Every configured stage scheduler must be instantiated on its stage."""
        settings = OptimizerPolicySettings(
            stages=(
                OptimizationStageSettings(
                    optimizer=AdamWSettings(),
                    scheduler=StepLRSettings(step_size=5, gamma=0.5),
                ),
                OptimizationStageSettings(
                    optimizer=AdamSettings(),
                    scheduler=ReduceLROnPlateauSettings(patience=2, factor=0.5),
                ),
            )
        )
        program = OptimizerPolicyBuilder().build(tiny_model, settings)

        assert isinstance(program.stages[0].scheduler, torch.optim.lr_scheduler.StepLR)
        assert isinstance(program.stages[1].scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_builder_concurrent_default_scheduler_exists_at_runtime(
        self, tiny_model: nn.Sequential
    ) -> None:
        """Concurrent single-stage programs must still materialize their scheduler."""
        settings = OptimizerPolicySettings(
            default_optimizer=ConcurrentOptimizerSettings(
                optimizers=(AdamWSettings(), AdamSettings()),
                selectors=(
                    ParameterSelectorSettings(prefix="0"),
                    ParameterSelectorSettings(prefix="1"),
                ),
            ),
            default_scheduler=StepLRSettings(step_size=5, gamma=0.5),
        )
        program = OptimizerPolicyBuilder().build(tiny_model, settings)

        stage = program.stages[0]
        assert isinstance(stage, ActiveStage)
        assert isinstance(stage.optimizer, ConcurrentOptimizer)
        assert isinstance(stage.scheduler, torch.optim.lr_scheduler.StepLR)

    def test_builder_module_path_selector_raises_when_layer_uncovered(
        self, tiny_model: nn.Sequential
    ) -> None:
        """Stage with a prefix selector that leaves some sub-module params uncovered raises.

        tiny_model is Sequential(Linear(4,8), Linear(8,2)).
        prefix="0" covers only layer-0 params; layer-1 params (1.weight, 1.bias) are
        unmatched, which is a training-correctness error.

        Args:
            tiny_model: Tiny two-layer model fixture.
        """
        settings = OptimizerPolicySettings(
            stages=(
                OptimizationStageSettings(
                    optimizer=AdamWSettings(),
                    selector=ParameterSelectorSettings(prefix="0"),
                ),
            )
        )
        with pytest.raises(ParameterPartitionError):
            OptimizerPolicyBuilder().build(tiny_model, settings)

    def test_builder_role_selector_unknown_raises_because_inference_assigns_all_roles(
        self, tiny_model: nn.Sequential
    ) -> None:
        """RoleSelectorSettings(role='unknown') raises when role inference assigns all params.

        With the default inference strategy active, standard Linear layers receive INPUT,
        HIDDEN, or OUTPUT roles — none remain UNKNOWN, so the UNKNOWN selector matches 0
        params while all 4 parameters are left unoptimized, which is a hard error.

        Args:
            tiny_model: Tiny two-layer model fixture.
        """
        settings = OptimizerPolicySettings(
            stages=(
                OptimizationStageSettings(
                    optimizer=AdamWSettings(),
                    selector=ParameterSelectorSettings(role="unknown"),
                ),
            )
        )
        with pytest.raises(ParameterPartitionError):
            OptimizerPolicyBuilder().build(tiny_model, settings)
