"""Optimization subsystem for neural network training.

This package provides infrastructure for multi-stage optimization including:
- Parameter inventory and role classification
- Parameter selector and partitioning logic
- Optimizer and scheduler factory patterns
- Stage transition triggers
- Execution layer: program state, stepping policies, controllers

Public API:
- IParameterInventory, TorchParameterInventory: parameter enumeration
- ParameterDescriptor: individual parameter metadata
- IParameterRoleInferenceStrategy: role classification interface
- CompositeParameterRoleInferenceStrategy, make_default_inference_strategy: role inference
- IParameterSelector: parameter filtering interface
- RoleSelector, ModulePathSelector, MuonEligibleSelector, etc.: concrete selectors
- IParameterPartitioner, ParameterPartitioner: parameter grouping
- IOptimizerFactory, TorchOptimizerFactory: optimizer instantiation
- ISchedulerFactory, TorchSchedulerFactory: scheduler instantiation
- ITransitionTrigger: stage transition control
- EpochTransitionTrigger, PlateauTransitionTrigger, NoTransitionTrigger: triggers
- BatchedMuon: drop-in torch.optim.Muon subclass with grouped bmm Newton-Schulz (faster)
- ConcurrentOptimizer: real torch.optim.Optimizer wrapping multiple sub-optimizers
- MuonMixedOptimizer: ConcurrentOptimizer pre-wired for Muon + AdamW companion split
- IManualOptimizationHost, IManualOptimizer: narrow protocols for Lightning-aware
  manual stepping without depending on concrete wrapper classes
- ActiveStage, RunningOptimizerPolicy: state objects
- IOptimizationStateRepository, OptimizationStateRepository: checkpoint management
- IStepPolicy, StepAllOptimizers, AlternatingStepPolicy, LBFGSStageStepper: stepping
- OptimizationMetricsView: read-only state projection
- IOptimizerPolicyBuilder, OptimizerPolicyBuilder: assembly
- IOptimizationController, AutomaticOptimizationController, ManualOptimizationController: control
"""

from __future__ import annotations

from .batched_muon import BatchedMuon
from .builder import (
    IOptimizerPolicyBuilder,
    OptimizerPolicyBuilder,
)
from .concurrent_optimizer import ConcurrentOptimizer, MuonMixedOptimizer
from .controllers import (
    AutomaticOptimizationController,
    IOptimizationController,
    ManualOptimizationController,
)
from .factories import (
    IMuonOptimizerFactory,
    IOptimizerFactory,
    ISchedulerFactory,
    TorchOptimizerFactory,
    TorchSchedulerFactory,
)
from .inventory import (
    IParameterInventory,
    ParameterDescriptor,
    TorchParameterInventory,
)
from .manual_host import IManualOptimizationHost, IManualOptimizer
from .metrics import OptimizationMetricsView
from .partitioning import (
    IParameterPartitioner,
    ParameterPartitioner,
)
from .role_inference import (
    CompositeParameterRoleInferenceStrategy,
    IParameterRoleInferenceStrategy,
    make_default_inference_strategy,
)
from .selectors import (
    DifferenceSelector,
    IntersectionSelector,
    IParameterSelector,
    ModulePathSelector,
    MuonEligibleSelector,
    NonMuonSelector,
    RoleSelector,
    UnionSelector,
)
from .state import (
    ActiveStage,
    RunningOptimizerPolicy,
)
from .state_repository import (
    IOptimizationStateRepository,
    OptimizationStateRepository,
)
from .stepping import (
    AlternatingStepPolicy,
    IStepPolicy,
    LBFGSStageStepper,
    StepAllOptimizers,
)
from .triggers import (
    EpochTransitionTrigger,
    ITransitionTrigger,
    NoTransitionTrigger,
    PlateauTransitionTrigger,
)

__all__ = [
    "IParameterInventory",
    "ParameterDescriptor",
    "TorchParameterInventory",
    "IParameterRoleInferenceStrategy",
    "CompositeParameterRoleInferenceStrategy",
    "make_default_inference_strategy",
    "IParameterSelector",
    "RoleSelector",
    "ModulePathSelector",
    "IntersectionSelector",
    "UnionSelector",
    "DifferenceSelector",
    "MuonEligibleSelector",
    "NonMuonSelector",
    "IParameterPartitioner",
    "ParameterPartitioner",
    "IOptimizerFactory",
    "ISchedulerFactory",
    "IMuonOptimizerFactory",
    "TorchOptimizerFactory",
    "TorchSchedulerFactory",
    "ITransitionTrigger",
    "EpochTransitionTrigger",
    "PlateauTransitionTrigger",
    "NoTransitionTrigger",
    "BatchedMuon",
    "ConcurrentOptimizer",
    "MuonMixedOptimizer",
    "ActiveStage",
    "RunningOptimizerPolicy",
    "IOptimizationStateRepository",
    "OptimizationStateRepository",
    "IStepPolicy",
    "StepAllOptimizers",
    "AlternatingStepPolicy",
    "LBFGSStageStepper",
    "OptimizationMetricsView",
    "IOptimizerPolicyBuilder",
    "OptimizerPolicyBuilder",
    "IOptimizationController",
    "AutomaticOptimizationController",
    "ManualOptimizationController",
    "IManualOptimizationHost",
    "IManualOptimizer",
]
