"""Builder for assembling RunningOptimizationProgram from model and config."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch.nn as nn

from dlkit.infrastructure.config.optimization_program import OptimizationProgramSettings
from dlkit.infrastructure.config.optimization_stage import (
    ConcurrentOptimizationSettings,
    OptimizationStageSettings,
)
from dlkit.infrastructure.config.optimization_trigger import TriggerSettings

from .factories import TorchOptimizerFactory, TorchSchedulerFactory
from .inventory import ParameterDescriptor, TorchParameterInventory
from .partitioning import ParameterPartitioner
from .selectors import IParameterSelector
from .state import ActiveConcurrentGroup, ActiveStage, RunningOptimizationProgram
from .triggers import (
    EpochTransitionTrigger,
    ITransitionTrigger,
    NoTransitionTrigger,
    PlateauTransitionTrigger,
)

# Parameter group as expected by torch.optim.Optimizer
type ParamGroup = dict[str, object]


class IOptimizationProgramBuilder(ABC):
    """Abstract interface for building optimization programs."""

    @abstractmethod
    def build(
        self,
        model: nn.Module,
        settings: OptimizationProgramSettings,
    ) -> RunningOptimizationProgram:
        """Build a RunningOptimizationProgram from a model and settings.

        Args:
            model: The neural network model.
            settings: Optimization program configuration.

        Returns:
            A RunningOptimizationProgram ready for execution.
        """
        ...


def _build_trigger(trigger_settings: TriggerSettings) -> ITransitionTrigger:
    """Build a trigger from settings.

    Args:
        trigger_settings: Trigger configuration (or None).

    Returns:
        An ITransitionTrigger instance.
    """
    if trigger_settings is None:
        return NoTransitionTrigger()

    if hasattr(trigger_settings, "at_epoch"):
        # EpochTriggerSettings
        at_epoch = getattr(trigger_settings, "at_epoch", 0)
        return EpochTransitionTrigger(at_epoch=int(at_epoch))

    if hasattr(trigger_settings, "monitor"):
        # PlateauTriggerSettings
        mode_str = str(getattr(trigger_settings, "mode", "min"))
        mode: Literal["min", "max"] = "min" if mode_str == "min" else "max"
        return PlateauTransitionTrigger(
            monitor=str(getattr(trigger_settings, "monitor", "val_loss")),
            patience=int(getattr(trigger_settings, "patience", 3)),
            min_delta=float(getattr(trigger_settings, "min_delta", 1e-4)),
            mode=mode,
        )

    return NoTransitionTrigger()


def _params_to_group(descriptors: tuple[ParameterDescriptor, ...]) -> list[ParamGroup]:
    """Convert parameter descriptors to optimizer param_groups format.

    Args:
        descriptors: Tuple of ParameterDescriptor objects.

    Returns:
        List with one param group containing all descriptors' parameters.
    """
    # Extract .parameter attribute from each descriptor
    params = [d.parameter for d in descriptors]
    return [{"params": params}]


class OptimizationProgramBuilder(IOptimizationProgramBuilder):
    """Concrete builder for optimization programs.

    Assembles a RunningOptimizationProgram by:
    1. Enumerating model parameters via TorchParameterInventory
    2. Partitioning parameters via selectors (or using all if no selector)
    3. Creating optimizers and schedulers via factory pattern
    4. Building triggers for stage transitions
    5. Wrapping in ActiveStage or ActiveConcurrentGroup objects
    """

    def build(
        self,
        model: nn.Module,
        settings: OptimizationProgramSettings,
    ) -> RunningOptimizationProgram:
        """Build the optimization program.

        Args:
            model: The neural network model.
            settings: Optimization program configuration.

        Returns:
            A RunningOptimizationProgram with stages initialized.
        """
        # Empty stages: use default_optimizer + default_scheduler as fallback
        if not settings.stages:
            inventory = TorchParameterInventory(model)
            all_params = inventory.list_parameters()

            param_groups = _params_to_group(all_params)

            optimizer = TorchOptimizerFactory(settings.default_optimizer).create(param_groups)

            scheduler = None
            if settings.default_scheduler is not None:
                scheduler = TorchSchedulerFactory(settings.default_scheduler).create(optimizer)

            trigger = NoTransitionTrigger()

            stage = ActiveStage(
                optimizer=optimizer,
                scheduler=scheduler,
                trigger=trigger,
                stage_index=0,
                name="default",
            )

            return RunningOptimizationProgram(stages=(stage,))

        # Multi-stage: build each stage from config
        stages: list[ActiveStage | ActiveConcurrentGroup] = []

        for idx, stage_config in enumerate(settings.stages):
            if isinstance(stage_config, OptimizationStageSettings):
                # Single optimizer stage
                stage = self._build_single_stage(model, stage_config, idx)
                stages.append(stage)

            elif isinstance(stage_config, ConcurrentOptimizationSettings):
                # Concurrent optimizers
                group = self._build_concurrent_group(model, stage_config, idx)
                stages.append(group)

        return RunningOptimizationProgram(stages=tuple(stages))

    def _build_single_stage(
        self, model: nn.Module, config: OptimizationStageSettings, stage_index: int
    ) -> ActiveStage:
        """Build a single optimizer stage.

        Args:
            model: The neural network model.
            config: Stage configuration.
            stage_index: Zero-based stage index.

        Returns:
            An ActiveStage instance.
        """
        # Enumerate parameters
        inventory = TorchParameterInventory(model)

        # Partition parameters if selector provided
        if config.selector is not None:
            selector = self._selector_from_settings(config.selector)
            partitioner = ParameterPartitioner()
            partitions = partitioner.partition(inventory, [selector])
            selected_params: tuple[ParameterDescriptor, ...] = partitions[0]
        else:
            # Use all parameters
            selected_params = inventory.list_parameters()

        # Build param groups
        param_groups = _params_to_group(selected_params)

        # Create optimizer
        optimizer = TorchOptimizerFactory(config.optimizer).create(param_groups)

        # Create scheduler if provided
        scheduler = None
        if config.scheduler is not None:
            scheduler = TorchSchedulerFactory(config.scheduler).create(optimizer)

        # Build trigger
        trigger = _build_trigger(config.trigger)

        monitor = config.scheduler.monitor if config.scheduler is not None else "val_loss"
        frequency = config.scheduler.frequency if config.scheduler is not None else 1

        return ActiveStage(
            optimizer=optimizer,
            scheduler=scheduler,
            trigger=trigger,
            stage_index=stage_index,
            name="",
            scheduler_monitor=monitor,
            scheduler_frequency=frequency,
        )

    def _build_concurrent_group(
        self, model: nn.Module, config: ConcurrentOptimizationSettings, group_index: int
    ) -> ActiveConcurrentGroup:
        """Build a concurrent optimizer group.

        Args:
            model: The neural network model.
            config: Concurrent group configuration.
            group_index: Zero-based group index.

        Returns:
            An ActiveConcurrentGroup instance.
        """
        stages: list[ActiveStage] = []

        for idx, stage_config in enumerate(config.optimizers):
            stage = self._build_single_stage(model, stage_config, stage_index=idx)
            stages.append(stage)

        # Build trigger for the group
        trigger = _build_trigger(config.trigger)

        return ActiveConcurrentGroup(
            stages=tuple(stages),
            trigger=trigger,
            group_index=group_index,
        )

    def _selector_from_settings(self, selector_settings: object) -> IParameterSelector:
        """Convert selector settings to an IParameterSelector.

        This is a placeholder that expects selector_settings to already be
        converted to an IParameterSelector elsewhere in the config pipeline.

        Args:
            selector_settings: Selector configuration object.

        Returns:
            An IParameterSelector instance.
        """
        # Direct passthrough if already an IParameterSelector
        if isinstance(selector_settings, IParameterSelector):
            return selector_settings

        # Otherwise raise error (selectors should be resolved earlier in pipeline)
        raise TypeError(
            f"Selector settings must be IParameterSelector, got {type(selector_settings)}"
        )
