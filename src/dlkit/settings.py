"""User-facing settings namespace.

Thin re-exports from ``dlkit.infrastructure.config`` so users can write::

    from dlkit.settings import GeneralSettings, TrainingSettings

instead of the internal path::

    from dlkit.infrastructure.config import GeneralSettings, TrainingSettings
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
    ExtrasSettings,
    FactoryProvider,
    FlowMatchingSettings,
    GeneralSettings,
    GenerativeSettings,
    HyperParameterSettings,
    IndexSplitSettings,
    InferenceWorkflowSettings,
    LossComponentSettings,
    MetricComponentSettings,
    MLflowSettings,
    ModelComponentSettings,
    OptimizerSettings,
    OptunaSettings,
    SchedulerSettings,
    SessionSettings,
    TrainerSettings,
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
    ContextFeature,
    DataEntry,
    Feature,
    FeatureType,
    Latent,
    SparseFeature,
    Target,
    TargetType,
)

__all__ = [
    # Data entry factories (for programmatic config)
    "Feature",
    "Target",
    "ContextFeature",
    "SparseFeature",
    "AutoencoderTarget",
    "Latent",
    "DataEntry",
    "FeatureType",
    "TargetType",
    # Top-level unified config
    "GeneralSettings",
    "SessionSettings",
    # Workflow-specific settings
    "BaseWorkflowSettings",
    "TrainingWorkflowSettings",
    "InferenceWorkflowSettings",
    # Functional settings
    "MLflowSettings",
    "OptunaSettings",
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
