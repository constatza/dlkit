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

Plugin System:
- PluginConfig: Base plugin configuration
- MLflowPluginConfig: MLflow experiment tracking
- OptunaPluginConfig: Hyperparameter optimization
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

# Partial loading factories
# Component settings
# Utility settings
from .dataloader_settings import DataloaderSettings
from .datamodule_settings import DataModuleSettings
from .dataset_settings import DatasetSettings, IndexSplitSettings
from .extras_settings import ExtrasSettings
from .factories import (
    WorkflowSettingsLoader,
    default_settings_loader,
    load_sections,
    load_settings,
)

# Main settings
from .general_settings import GeneralSettings

# Generative algorithm settings
from .generative_settings import CNFSettings, FlowMatchingSettings, GenerativeSettings

# Flattened functional settings
from .mlflow_settings import MLflowSettings
from .model_components import (
    LossComponentSettings,
    MetricComponentSettings,
    ModelComponentSettings,
    WrapperComponentSettings,
)
from .optimization_program import OptimizationProgramSettings
from .optimization_selector import ParameterSelectorSettings
from .optimization_stage import (
    ConcurrentOptimizationSettings,
    OptimizationStageSettings,
)
from .optimization_trigger import (
    EpochTriggerSettings,
    PlateauTriggerSettings,
    TriggerSettings,
)
from .optimizer_component import (
    OptimizerComponentSettings,
    SchedulerComponentSettings,
)
from .optimizer_settings import OptimizerSettings, SchedulerSettings
from .optuna_settings import OptunaSettings
from .session_settings import SessionSettings

# External library settings (unchanged - kept compact as requested)
from .trainer_settings import TrainerSettings
from .training_settings import TrainingSettings
from .transform_settings import TransformSettings

# Workflow-specific settings (SOLID-compliant)
from .workflow_settings import (
    BaseWorkflowSettings,
    InferenceWorkflowSettings,
    TrainingWorkflowSettings,
)

__all__ = [
    # Main settings
    "GeneralSettings",
    "SessionSettings",
    # Workflow-specific settings (SOLID-compliant)
    "BaseWorkflowSettings",
    "TrainingWorkflowSettings",
    "InferenceWorkflowSettings",
    # Partial loading factories
    "WorkflowSettingsLoader",
    "default_settings_loader",
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
    # Optimizer and scheduler components
    "OptimizerComponentSettings",
    "SchedulerComponentSettings",
    # Optimization configuration
    "OptimizationProgramSettings",
    "OptimizationStageSettings",
    "ConcurrentOptimizationSettings",
    "ParameterSelectorSettings",
    "EpochTriggerSettings",
    "PlateauTriggerSettings",
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
