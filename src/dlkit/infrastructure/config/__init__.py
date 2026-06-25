"""DLKit Settings - SOLID-compliant configuration system.

This module provides a comprehensive settings system for DLKit with:
- SOLID principle compliance
- Factory pattern for object construction
- Flattened hierarchy for maintainability

Main Classes:
- JobConfig: Top-level job configuration (base)
- TrainingJobConfig: Validated training job
- InferenceJobConfig: Validated inference job
- SearchJobConfig: Validated HPO job
"""

# Core infrastructure
from .core import (
    BasicSettings,
    BuildContext,
    ComponentFactory,
    ComponentRegistry,
    ComponentSettings,
    FactoryProvider,
    HyperParameterSettings,
)
from .core.updater import update_settings

# New settings classes
from .data_settings import DataModuleSelector, DataSettings

# Partial loading factories
# Component settings
# Utility settings
from .dataloader_settings import DataloaderSettings
from .experiment_settings import ExperimentSettings
from .extras_settings import ExtrasSettings
from .factories import (
    WorkflowSettingsLoader,
    default_settings_loader,
    load_job,
    load_sections,
    load_settings,
)

# Generative algorithm settings
from .generative_settings import CNFSettings, FlowMatchingSettings, GenerativeSettings

# Job config (new top-level API)
from .job_config import (
    InferenceJobConfig,
    JobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)
from .model_components import (
    CrossEntropyLossSettings,
    HuberLossSettings,
    L1LossSettings,
    LossComponentSettings,
    LossSpec,
    MeanAbsoluteErrorSettings,
    MeanSquaredErrorSettings,
    MetricComponentSettings,
    MetricSpec,
    ModelComponentSettings,
    MSELossSettings,
    R2ScoreSettings,
    WrapperComponentSettings,
)
from .model_settings import ModelParams, ModelSettings
from .optimization_selector import ParameterSelectorSettings
from .optimization_stage import OptimizationStageSettings
from .optimization_trigger import TriggerSettings
from .optimizer_component import (
    AdamSettings,
    AdamWSettings,
    BatchedMuonSettings,
    ConcurrentOptimizerSettings,
    CosineAnnealingLRSettings,
    CosineAnnealingWarmRestartsSettings,
    LBFGSSettings,
    MuonSettings,
    OptimizerComponentSettings,
    OptimizerSpec,
    ReduceLROnPlateauSettings,
    SchedulerComponentSettings,
    SchedulerSpec,
    StepLRSettings,
)
from .optimizer_policy import OptimizerPolicySettings
from .optimizer_settings import OptimizerSettings, SchedulerSettings
from .run_settings import RunSettings, RunType
from .search_settings import (
    CategoricalParam,
    ConstantParam,
    FloatParam,
    IntParam,
    LogFloatParam,
    LogIntParam,
    PrunerSettings,
    SamplerSettings,
    SearchSettings,
    SpaceParam,
)
from .tracking_settings import TrackingSettings

# External library settings (unchanged - kept compact as requested)
from .trainer_settings import TrainerSettings
from .training_settings import StoppingSettings, TrainingSettings
from .transform_settings import TransformSettings

__all__ = [
    # Job config (new top-level API)
    "JobConfig",
    "TrainingJobConfig",
    "InferenceJobConfig",
    "SearchJobConfig",
    # New settings classes
    "RunSettings",
    "RunType",
    "ExperimentSettings",
    "ModelSettings",
    "ModelParams",
    "DataSettings",
    "DataModuleSelector",
    "TrainingSettings",
    "StoppingSettings",
    "SearchSettings",
    "SpaceParam",
    "FloatParam",
    "LogFloatParam",
    "IntParam",
    "LogIntParam",
    "CategoricalParam",
    "ConstantParam",
    "SamplerSettings",
    "PrunerSettings",
    "TrackingSettings",
    # Partial loading factories
    "WorkflowSettingsLoader",
    "default_settings_loader",
    "load_job",
    "load_settings",
    "load_sections",
    # Core infrastructure
    "BasicSettings",
    "BuildContext",
    "ComponentFactory",
    "ComponentRegistry",
    "ComponentSettings",
    "FactoryProvider",
    "HyperParameterSettings",
    "update_settings",
    # Component settings
    "ModelComponentSettings",
    "MetricComponentSettings",
    "LossComponentSettings",
    "WrapperComponentSettings",
    # Typed loss function settings
    "MSELossSettings",
    "L1LossSettings",
    "HuberLossSettings",
    "CrossEntropyLossSettings",
    "LossSpec",
    # Typed metric settings
    "MeanSquaredErrorSettings",
    "MeanAbsoluteErrorSettings",
    "R2ScoreSettings",
    "MetricSpec",
    # Optimizer and scheduler components
    "AdamSettings",
    "AdamWSettings",
    "BatchedMuonSettings",
    "ConcurrentOptimizerSettings",
    "LBFGSSettings",
    "MuonSettings",
    "OptimizerComponentSettings",
    "OptimizerSpec",
    "CosineAnnealingLRSettings",
    "CosineAnnealingWarmRestartsSettings",
    "ReduceLROnPlateauSettings",
    "SchedulerComponentSettings",
    "SchedulerSpec",
    "StepLRSettings",
    # Optimization configuration
    "OptimizerPolicySettings",
    "OptimizationStageSettings",
    # Selector
    "ParameterSelectorSettings",
    # Trigger
    "TriggerSettings",
    # External library settings
    "TrainerSettings",
    "OptimizerSettings",
    "SchedulerSettings",
    # Utility settings
    "DataloaderSettings",
    "TransformSettings",
    "ExtrasSettings",
    # Generative algorithm settings
    "GenerativeSettings",
    "FlowMatchingSettings",
    "CNFSettings",
]
