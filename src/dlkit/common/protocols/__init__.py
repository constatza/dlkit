"""Framework-agnostic structural protocols for shared contracts."""

from .model_protocols import IDataModule, ITrainableModule
from .settings_protocols import (
    BaseSettingsProtocol,
    ModelSettingsProtocol,
    TrainingSettingsProtocol,
)

__all__ = [
    "IDataModule",
    "ITrainableModule",
    "BaseSettingsProtocol",
    "ModelSettingsProtocol",
    "TrainingSettingsProtocol",
]
