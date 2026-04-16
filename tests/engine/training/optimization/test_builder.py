"""Tests for the OptimizationProgramBuilder."""

from __future__ import annotations

import torch
import torch.nn as nn

from dlkit.engine.training.optimization.builder import OptimizationProgramBuilder
from dlkit.engine.training.optimization.state import ActiveStage
from dlkit.engine.training.optimization.triggers import EpochTransitionTrigger
from dlkit.infrastructure.config.optimization_program import OptimizationProgramSettings
from dlkit.infrastructure.config.optimization_stage import OptimizationStageSettings
from dlkit.infrastructure.config.optimization_trigger import EpochTriggerSettings
from dlkit.infrastructure.config.optimizer_component import OptimizerComponentSettings


class TestOptimizationProgramBuilder:
    """Tests for OptimizationProgramBuilder."""

    def test_builder_degenerate_single_stage(
        self,
        tiny_model: nn.Sequential,
    ) -> None:
        """Verify builder creates single stage from empty settings.

        Args:
            tiny_model: Tiny model fixture.
        """
        # Empty stages tuple -> defaults to AdamW optimizer
        settings = OptimizationProgramSettings()
        builder = OptimizationProgramBuilder()

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
            optimizer=OptimizerComponentSettings(name="SGD"),
        )
        settings = OptimizationProgramSettings(stages=(stage_config,))
        builder = OptimizationProgramBuilder()

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
            optimizer=OptimizerComponentSettings(name="SGD"),
        )
        stage_1_config = OptimizationStageSettings(
            optimizer=OptimizerComponentSettings(name="Adam"),
        )
        settings = OptimizationProgramSettings(
            stages=(stage_0_config, stage_1_config),
        )
        builder = OptimizationProgramBuilder()

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
            optimizer=OptimizerComponentSettings(name="SGD"),
            trigger=trigger_config,
        )
        settings = OptimizationProgramSettings(stages=(stage_config,))
        builder = OptimizationProgramBuilder()

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
        settings = OptimizationProgramSettings()
        builder = OptimizationProgramBuilder()

        program = builder.build(tiny_model, settings)

        stage_entry = program.stages[0]
        assert isinstance(stage_entry, ActiveStage)
        # Verify it's an Adam optimizer (AdamW is a variant of Adam)
        assert isinstance(stage_entry.optimizer, torch.optim.AdamW)
