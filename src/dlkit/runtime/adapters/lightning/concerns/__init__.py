"""Extracted concerns for the ProcessingLightningWrapper architecture.

This package separates cross-cutting concerns from the base wrapper:
- IStepLogger, LightningStepLogger, NullStepLogger: Metric logging
- ICheckpointSerializer, DLKitCheckpointSerializer: Checkpoint persistence
- ILearningRateManager, ConfigLearningRateManager: Learning rate management
"""

from dlkit.runtime.adapters.lightning.concerns.checkpoint_serializer import (
    DLKitCheckpointSerializer,
    ICheckpointSerializer,
)
from dlkit.runtime.adapters.lightning.concerns.lr_manager import (
    ConfigLearningRateManager,
    ILearningRateManager,
)
from dlkit.runtime.adapters.lightning.concerns.step_logger import (
    IStepLogger,
    LightningStepLogger,
    NullStepLogger,
)

__all__ = [
    "ConfigLearningRateManager",
    "DLKitCheckpointSerializer",
    "ICheckpointSerializer",
    "ILearningRateManager",
    "IStepLogger",
    "LightningStepLogger",
    "NullStepLogger",
]
