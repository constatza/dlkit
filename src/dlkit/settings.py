"""User-facing settings namespace.

Thin re-exports from ``dlkit.infrastructure.config`` so users can write::

    from dlkit.settings import TrainingSettings

instead of the internal path::

    from dlkit.infrastructure.config import TrainingSettings
"""

from dlkit.infrastructure.config import (
    BasicSettings,
    BuildContext,
    CategoricalParam,
    CNFSettings,
    ComponentFactory,
    ComponentRegistry,
    ComponentSettings,
    ConstantParam,
    DataloaderSettings,
    DataModuleSelector,
    DataSettings,
    ExperimentSettings,
    ExtrasSettings,
    FactoryProvider,
    FloatParam,
    FlowMatchingSettings,
    GenerativeSettings,
    HyperParameterSettings,
    InferenceJobConfig,
    IntParam,
    JobConfig,
    LogFloatParam,
    LogIntParam,
    LossComponentSettings,
    MetricComponentSettings,
    ModelComponentSettings,
    ModelParams,
    ModelSettings,
    OptimizerSettings,
    PrunerSettings,
    RunSettings,
    SamplerSettings,
    SchedulerSettings,
    SearchJobConfig,
    SearchSettings,
    SpaceParam,
    StoppingSettings,
    TrackingSettings,
    TrainerSettings,
    TrainingJobConfig,
    TrainingSettings,
    TransformSettings,
    WorkflowSettingsLoader,
    WrapperComponentSettings,
    default_settings_loader,
    load_job,
    load_sections,
    load_settings,
    update_settings,
)
from dlkit.infrastructure.config.data_entries import (
    AutoencoderTarget,
    DataEntry,
    Latent,
)

__all__ = [
    # Data entry types
    "AutoencoderTarget",
    "Latent",
    "DataEntry",
    # Job config (new top-level API)
    "JobConfig",
    "TrainingJobConfig",
    "InferenceJobConfig",
    "SearchJobConfig",
    # New settings classes
    "RunSettings",
    "ExperimentSettings",
    "ModelSettings",
    "ModelParams",
    "DataSettings",
    "DataModuleSelector",
    "TrackingSettings",
    "SearchSettings",
    "StoppingSettings",
    # HPO search space param types
    "SpaceParam",
    "FloatParam",
    "LogFloatParam",
    "IntParam",
    "LogIntParam",
    "CategoricalParam",
    "ConstantParam",
    "SamplerSettings",
    "PrunerSettings",
    # Functional settings
    "TrainingSettings",
    "DataloaderSettings",
    "TransformSettings",
    "ExtrasSettings",
    # Component settings
    "ModelComponentSettings",
    "MetricComponentSettings",
    "LossComponentSettings",
    "WrapperComponentSettings",
    # External library settings
    "TrainerSettings",
    "OptimizerSettings",
    "SchedulerSettings",
    # Generative algorithm settings
    "GenerativeSettings",
    "FlowMatchingSettings",
    "CNFSettings",
    # Core infrastructure / base classes
    "BasicSettings",
    "ComponentSettings",
    "HyperParameterSettings",
    "BuildContext",
    "ComponentFactory",
    "ComponentRegistry",
    "FactoryProvider",
    # Loading utilities
    "WorkflowSettingsLoader",
    "default_settings_loader",
    "load_job",
    "load_settings",
    "load_sections",
    "update_settings",
]
