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
- ActiveStage, ActiveConcurrentGroup, RunningOptimizationProgram: state objects
- IOptimizationStateRepository, OptimizationStateRepository: checkpoint management
- IStepPolicy, StepAllOptimizers, AlternatingStepPolicy, LBFGSStageStepper: stepping
- OptimizationMetricsView: read-only state projection
- IOptimizationProgramBuilder, OptimizationProgramBuilder: assembly
- IOptimizationController, AutomaticOptimizationController, ManualOptimizationController: control
"""

from __future__ import annotations

from .builder import (
    IOptimizationProgramBuilder,
    OptimizationProgramBuilder,
)
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
    ActiveConcurrentGroup,
    ActiveStage,
    RunningOptimizationProgram,
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
    "ActiveStage",
    "ActiveConcurrentGroup",
    "RunningOptimizationProgram",
    "IOptimizationStateRepository",
    "OptimizationStateRepository",
    "IStepPolicy",
    "StepAllOptimizers",
    "AlternatingStepPolicy",
    "LBFGSStageStepper",
    "OptimizationMetricsView",
    "IOptimizationProgramBuilder",
    "OptimizationProgramBuilder",
    "IOptimizationController",
    "AutomaticOptimizationController",
    "ManualOptimizationController",
]
