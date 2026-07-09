"""Extracted concerns for the ProcessingLightningWrapper architecture.

This package separates cross-cutting concerns from the base wrapper:
- LightningStepLogger: Metric logging
- ICheckpointSerializer, DLKitCheckpointSerializer: Checkpoint persistence
- ILearningRateManager, ConfigLearningRateManager: Learning rate management
"""

from dlkit.engine.adapters.lightning.concerns.checkpoint_serializer import (
    DLKitCheckpointSerializer,
    ICheckpointSerializer,
)
from dlkit.engine.adapters.lightning.concerns.lr_manager import (
    ConfigLearningRateManager,
    ILearningRateManager,
)
from dlkit.engine.adapters.lightning.concerns.step_logger import (
    LightningStepLogger,
)

__all__ = [
    "ConfigLearningRateManager",
    "DLKitCheckpointSerializer",
    "ICheckpointSerializer",
    "ILearningRateManager",
    "LightningStepLogger",
]
