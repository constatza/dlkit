"""DLKit Settings - SOLID-compliant configuration system.

This module provides a comprehensive settings system for DLKit with:
- SOLID principle compliance
- Factory pattern for object construction
- Mode separation (training vs inference)
- Plugin architecture for optional features
- Flattened hierarchy for maintainability

Main Classes:
- GeneralSettings: Top-level settings with mode separation
- SessionSettings: Session mode control and configuration
- TrainingModeSettings: Full training pipeline with plugins
- InferenceModeSettings: Lightweight inference configuration

Factory System:
- ComponentFactory: Abstract factory for component creation
- FactoryProvider: Global factory registry and access point
- BuildContext: Dependency injection context

Plugin System:
- PluginConfig: Base plugin configuration
- MLflowPluginConfig: MLflow experiment tracking
- OptunaPluginConfig: Hyperparameter optimization
"""

# Core infrastructure
from .core import (
    BasicSettings,
    ComponentSettings,
    HyperParameterSettings,
    ComponentFactory,
    ComponentRegistry,
    FactoryProvider,
    BuildContext,
)
from .core.updater import update_settings

# Main settings
from .general_settings import GeneralSettings
from .session_settings import SessionSettings

# Workflow-specific settings (SOLID-compliant)
from .workflow_settings import (
    BaseWorkflowSettings,
    TrainingWorkflowSettings,
    InferenceWorkflowSettings,
)

# Backward-compat aliases
BaseSettings = BaseWorkflowSettings
WorkflowTrainingSettings = TrainingWorkflowSettings
InferenceSettings = InferenceWorkflowSettings


# Partial loading factories
from .factories import (
    WorkflowSettingsLoader,
    PartialSettingsLoader,
    default_settings_loader,
    load_settings,
    load_sections,
)

# Flattened functional settings
from .mlflow_settings import MLflowSettings
from .optuna_settings import OptunaSettings
from .datamodule_settings import DataModuleSettings
from .dataset_settings import DatasetSettings, IndexSplitSettings
from .training_settings import TrainingSettings

# Component settings
from .components import (
    ModelComponentSettings,
    MetricComponentSettings,
    LossComponentSettings,
    WrapperComponentSettings,
)

# External library settings (unchanged - kept compact as requested)
from .trainer_settings import TrainerSettings
from .optimizer_settings import OptimizerSettings, SchedulerSettings

# Utility settings
from .dataloader_settings import DataloaderSettings
from .transform_settings import TransformSettings
from .extras_settings import ExtrasSettings

__all__ = [
    # Main settings
    "GeneralSettings",
    "SessionSettings",
    # Workflow-specific settings (SOLID-compliant)
    "BaseWorkflowSettings",
    "TrainingWorkflowSettings",
    "InferenceWorkflowSettings",
    # Backward-compat aliases
    "BaseSettings",
    "WorkflowTrainingSettings",
    "InferenceSettings",
    # Partial loading factories
    "WorkflowSettingsLoader",
    "PartialSettingsLoader",
    "default_settings_loader",
    "load_settings",
    "load_sections",
    # Core infrastructure
    "BasicSettings",
    "ComponentSettings",
    "HyperParameterSettings",
    "ComponentFactory",
    "ComponentRegistry",
    "FactoryProvider",
    "BuildContext",
    "update_settings",
    # Flattened functional settings
    "MLflowSettings",
    "OptunaSettings",
    "DataModuleSettings",
    "DatasetSettings",
    "IndexSplitSettings",
    "TrainingSettings",
    # Component settings
    "ModelComponentSettings",
    "MetricComponentSettings",
    "LossComponentSettings",
    "WrapperComponentSettings",
    # External library settings
    "TrainerSettings",
    "OptimizerSettings",
    "SchedulerSettings",
    # Utility settings
    "DataloaderSettings",
    "TransformSettings",
    "ExtrasSettings",
]
