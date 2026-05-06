"""Builder for assembling RunningOptimizerPolicy from model and config."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch.nn as nn
import torch.optim

_logger = logging.getLogger(__name__)

from dlkit.common.errors import ParameterPartitionError
from dlkit.domain.nn.parameter_roles import ParameterRole
from dlkit.infrastructure.config.optimization_selector import (
    ParameterSelectorSettings,
)
from dlkit.infrastructure.config.optimization_stage import OptimizationStageSettings
from dlkit.infrastructure.config.optimization_trigger import TriggerSettings
from dlkit.infrastructure.config.optimizer_component import (
    ConcurrentOptimizerSettings,
    MuonSettings,
)
from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings

from .concurrent_optimizer import ConcurrentOptimizer, MuonMixedOptimizer
from .factories import TorchOptimizerFactory, TorchSchedulerFactory
from .inventory import ParameterDescriptor, TorchParameterInventory
from .partitioning import ParameterPartitioner
from .role_inference import make_default_inference_strategy
from .selectors import (
    IParameterSelector,
    ModulePathSelector,
    MuonEligibleSelector,
    NonMuonSelector,
    RoleSelector,
)
from .state import ActiveStage, RunningOptimizerPolicy
from .triggers import (
    EpochTransitionTrigger,
    ITransitionTrigger,
    NoTransitionTrigger,
    PlateauTransitionTrigger,
)

# Parameter group as expected by torch.optim.Optimizer
type ParamGroup = dict[str, object]


class IOptimizerPolicyBuilder(ABC):
    """Abstract interface for building optimization programs."""

    @abstractmethod
    def build(
        self,
        model: nn.Module,
        settings: OptimizerPolicySettings,
    ) -> RunningOptimizerPolicy:
        """Build a RunningOptimizerPolicy from a model and settings.

        Args:
            model: The neural network model.
            settings: Optimization program configuration.

        Returns:
            A RunningOptimizerPolicy ready for execution.
        """
        ...


def _build_trigger(trigger_settings: TriggerSettings | None) -> ITransitionTrigger:
    """Build a transition trigger from settings.

    Args:
        trigger_settings: Trigger configuration (or None for no transition).

    Returns:
        An ITransitionTrigger instance.
    """
    if trigger_settings is None:
        return NoTransitionTrigger()
    if trigger_settings.at_epoch is not None:
        return EpochTransitionTrigger(at_epoch=trigger_settings.at_epoch)
    if trigger_settings.patience is not None:
        return PlateauTransitionTrigger(
            monitor=trigger_settings.monitor,
            patience=trigger_settings.patience,
            min_delta=trigger_settings.min_delta,
            mode=trigger_settings.mode,
        )
    return NoTransitionTrigger()


def _params_to_group(descriptors: tuple[ParameterDescriptor, ...]) -> list[ParamGroup]:
    """Convert parameter descriptors to optimizer param_groups format.

    Args:
        descriptors: Tuple of ParameterDescriptor objects.

    Returns:
        List with one param group containing all descriptors' parameters.
    """
    params = [d.parameter for d in descriptors]
    return [{"params": params}]


def _selector_from_settings(selector_settings: ParameterSelectorSettings) -> IParameterSelector:
    """Convert a ParameterSelectorSettings instance to an IParameterSelector.

    Args:
        selector_settings: Selector settings with role or prefix set.

    Returns:
        The corresponding IParameterSelector instance.

    Raises:
        ValueError: If the settings have neither role nor prefix set.
    """
    if selector_settings.role is not None:
        return RoleSelector(ParameterRole[selector_settings.role.upper()])
    if selector_settings.prefix is not None:
        return ModulePathSelector(selector_settings.prefix)
    raise ValueError("ParameterSelectorSettings has neither role nor prefix set.")


def _make_inventory(model: nn.Module) -> TorchParameterInventory:
    """Build a parameter inventory with default role inference for a model.

    Args:
        model: The neural network model.

    Returns:
        TorchParameterInventory with role resolver applied.
    """
    strategy = make_default_inference_strategy(model)
    return TorchParameterInventory(
        model,
        role_resolver=lambda d: strategy.infer(model, d.name, d.parameter) or d.role,
    )


class OptimizerPolicyBuilder(IOptimizerPolicyBuilder):
    """Concrete builder for optimization programs.

    Assembles a RunningOptimizerPolicy by:
    1. Enumerating model parameters via TorchParameterInventory
    2. Partitioning parameters via selectors (or using all if no selector)
    3. Creating optimizers and schedulers via factory pattern
    4. Building triggers for stage transitions
    5. Wrapping in ActiveStage objects (ConcurrentOptimizer handles concurrent stages)
    """

    def build(
        self,
        model: nn.Module,
        settings: OptimizerPolicySettings,
    ) -> RunningOptimizerPolicy:
        """Build the optimization program.

        Args:
            model: The neural network model.
            settings: Optimization program configuration.

        Returns:
            A RunningOptimizerPolicy with stages initialized.
        """
        if not settings.stages:
            return self._build_default(model, settings)

        stages: list[ActiveStage] = []
        for idx, stage_config in enumerate(settings.stages):
            stages.append(self._build_stage(model, stage_config, idx))
        return RunningOptimizerPolicy(stages=tuple(stages))

    def _build_default(
        self, model: nn.Module, settings: OptimizerPolicySettings
    ) -> RunningOptimizerPolicy:
        """Build the single-stage default program from default_optimizer/scheduler.

        Args:
            model: The neural network model.
            settings: Policy settings (stages must be empty).

        Returns:
            A RunningOptimizerPolicy with one default stage.
        """
        if isinstance(settings.default_optimizer, ConcurrentOptimizerSettings):
            optimizer: torch.optim.Optimizer = self._build_concurrent_optimizer(
                model, settings.default_optimizer
            )
        elif isinstance(settings.default_optimizer, MuonSettings):
            optimizer = self._build_muon_mixed(model, settings.default_optimizer)
        else:
            inventory = _make_inventory(model)
            param_groups = _params_to_group(inventory.list_parameters())
            optimizer = TorchOptimizerFactory(settings.default_optimizer).create(param_groups)

        scheduler = None
        if settings.default_scheduler is not None:
            scheduler = TorchSchedulerFactory(settings.default_scheduler).create(optimizer)

        stage = ActiveStage(
            optimizer=optimizer,
            scheduler=scheduler,
            trigger=NoTransitionTrigger(),
            stage_index=0,
            name="default",
        )
        return RunningOptimizerPolicy(stages=(stage,))

    def _build_stage(
        self,
        model: nn.Module,
        config: OptimizationStageSettings,
        stage_index: int,
    ) -> ActiveStage:
        """Build a single optimization stage.

        Args:
            model: The neural network model.
            config: Stage configuration.
            stage_index: Zero-based stage index.

        Returns:
            An ActiveStage instance.
        """
        trigger = _build_trigger(config.trigger)
        monitor = config.scheduler.monitor if config.scheduler is not None else "val_loss"
        frequency = config.scheduler.frequency if config.scheduler is not None else 1

        if isinstance(config.optimizer, ConcurrentOptimizerSettings):
            optimizer: torch.optim.Optimizer = self._build_concurrent_optimizer(
                model, config.optimizer
            )
        else:
            inventory = _make_inventory(model)
            if config.selector is not None:
                selector = _selector_from_settings(config.selector)
                partitioner = ParameterPartitioner()
                partitions = partitioner.partition(inventory, [selector])
                selected_params = partitions[0]
            else:
                selected_params = inventory.list_parameters()

            param_groups = _params_to_group(selected_params)
            optimizer = TorchOptimizerFactory(config.optimizer).create(param_groups)

        scheduler = None
        if config.scheduler is not None:
            scheduler = TorchSchedulerFactory(config.scheduler).create(optimizer)

        return ActiveStage(
            optimizer=optimizer,
            scheduler=scheduler,
            trigger=trigger,
            stage_index=stage_index,
            name="",
            scheduler_monitor=monitor,
            scheduler_frequency=frequency,
        )

    def _build_muon_mixed(self, model: nn.Module, settings: MuonSettings) -> torch.optim.Optimizer:
        """Build a MuonMixedOptimizer when Muon is the sole default_optimizer.

        Per the official PyTorch Muon documentation, Muon only supports 2D hidden-layer
        weight matrices.  This method partitions the model's parameters using
        ``MuonEligibleSelector`` (2D HIDDEN) and ``NonMuonSelector`` (everything else),
        creates a Muon sub-optimizer for the eligible set, and pairs it with a companion
        AdamW sub-optimizer (same learning rate) for the remaining parameters.

        If all parameters happen to be Muon-eligible, a plain Muon optimizer is returned
        and no companion is created.

        Args:
            model: The neural network model.
            settings: Muon optimizer configuration.

        Returns:
            A ``MuonMixedOptimizer`` wrapping Muon + AdamW when non-eligible parameters
            exist, or a plain Muon optimizer when every parameter is eligible.
        """
        inventory = _make_inventory(model)
        partitioner = ParameterPartitioner()
        muon_params, fallback_params = partitioner.partition(
            inventory,
            [MuonEligibleSelector(), NonMuonSelector()],
            warn_unmatched=False,
        )

        muon_opt = TorchOptimizerFactory(settings).create(_params_to_group(muon_params))

        if not fallback_params:
            return muon_opt

        _logger.warning(
            "Muon auto-split: %d eligible params → Muon, %d non-eligible params → AdamW "
            "(lr=%s). Use ConcurrentOptimizerSettings to configure the companion optimizer.",
            len(muon_params),
            len(fallback_params),
            settings.lr,
        )
        companion_opt = torch.optim.AdamW([d.parameter for d in fallback_params], lr=settings.lr)
        return MuonMixedOptimizer(muon_opt, companion_opt)

    def _build_concurrent_optimizer(
        self,
        model: nn.Module,
        config: ConcurrentOptimizerSettings,
    ) -> ConcurrentOptimizer:
        """Build a ConcurrentOptimizer from a ConcurrentOptimizerSettings.

        When ``config.selectors`` is empty and any optimizer is MuonSettings,
        assigns MuonEligibleSelector to Muon and NonMuonSelector to all others.
        Otherwise uses the provided selectors (None = all parameters).

        Args:
            model: The neural network model.
            config: Concurrent optimizer configuration.

        Returns:
            A ConcurrentOptimizer wrapping per-sub-optimizer instances.
        """
        inventory = _make_inventory(model)

        selectors: list[IParameterSelector | None]
        if config.selectors:
            selectors = [
                _selector_from_settings(s) if s is not None else None for s in config.selectors
            ]
        else:
            has_muon = any(isinstance(opt, MuonSettings) for opt in config.optimizers)
            if has_muon:
                selectors = [
                    MuonEligibleSelector() if isinstance(opt, MuonSettings) else NonMuonSelector()
                    for opt in config.optimizers
                ]
            else:
                selectors = [None] * len(config.optimizers)

        sub_optimizers: list[torch.optim.Optimizer] = []
        for opt_config, selector in zip(config.optimizers, selectors, strict=True):
            if selector is not None:
                partitioner = ParameterPartitioner()
                partitions = partitioner.partition(inventory, [selector], warn_unmatched=False)
                selected_params = partitions[0]
            else:
                selected_params = inventory.list_parameters()

            param_groups = _params_to_group(selected_params)
            sub_optimizers.append(TorchOptimizerFactory(opt_config).create(param_groups))

        all_param_ids = {id(p) for p in model.parameters()}
        covered_ids = {
            id(p) for opt in sub_optimizers for pg in opt.param_groups for p in pg["params"]
        }
        uncovered = all_param_ids - covered_ids
        if uncovered:
            raise ParameterPartitionError(
                message=f"ConcurrentOptimizer: {len(uncovered)} parameter(s) are not covered "
                "by any sub-optimizer and will not be optimized",
                context={"uncovered_count": len(uncovered)},
            )

        return ConcurrentOptimizer(sub_optimizers)
