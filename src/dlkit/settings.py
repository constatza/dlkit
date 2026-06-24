"""User-facing settings namespace.

Thin re-exports from ``dlkit.infrastructure.config`` so users can write::

    from dlkit.settings import TrainingSettings

instead of the internal path::

    from dlkit.infrastructure.config import TrainingSettings
"""

from dlkit.infrastructure.config import (
    BaseWorkflowSettings,
    BasicSettings,
    BuildContext,
    CNFSettings,
    ComponentFactory,
    ComponentRegistry,
    ComponentSettings,
    DataloaderSettings,
    DataModuleSettings,
    DatasetSettings,
    DataSettings,
    ExperimentSettings,
    ExtrasSettings,
    FactoryProvider,
    FlowMatchingSettings,
    GenerativeSettings,
    HyperParameterSettings,
    IndexSplitSettings,
    InferenceJobConfig,
    InferenceWorkflowSettings,
    JobConfig,
    LossComponentSettings,
    MetricComponentSettings,
    ModelComponentSettings,
    ModelSettings,
    OptimizerSettings,
    RunSettings,
    SchedulerSettings,
    SearchJobConfig,
    SearchSettings,
    StoppingSettings,
    TrackingSettings,
    TrainerSettings,
    TrainingJobConfig,
    TrainingSettings,
    TrainingWorkflowSettings,
    TransformSettings,
    WorkflowSettingsLoader,
    WrapperComponentSettings,
    default_settings_loader,
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
    "DataSettings",
    "TrackingSettings",
    "SearchSettings",
    "StoppingSettings",
    # Workflow-specific settings
    "BaseWorkflowSettings",
    "TrainingWorkflowSettings",
    "InferenceWorkflowSettings",
    # Functional settings
    "TrainingSettings",
    "DatasetSettings",
    "DataModuleSettings",
    "IndexSplitSettings",
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
    "load_settings",
    "load_sections",
    "update_settings",
]
