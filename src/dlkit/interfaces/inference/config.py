"""Configuration dataclasses for inference subsystem.

Simplified dataclasses without state machines or complex hierarchies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from dlkit.tools.config.precision.strategy import PrecisionStrategy

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from dlkit.core.shape_specs.simple_inference import ShapeSummary
    from dlkit.core.training.transforms.chain import TransformChain


@dataclass
class PredictorConfig:
    """Configuration for predictor creation.

    Simple dataclass holding predictor configuration without
    complex validation or state management.
    """

    checkpoint_path: Path
    device: str = "auto"
    batch_size: int = 32
    apply_transforms: bool = True
    auto_load: bool = True
    precision: PrecisionStrategy | None = None

    def __post_init__(self):
        """Ensure checkpoint_path is a Path object."""
        if isinstance(self.checkpoint_path, str):
            self.checkpoint_path = Path(self.checkpoint_path)


@dataclass
class ModelState:
    """Model state container.

    Simplified from previous version - no state machine, just data.
    Holds loaded model and associated metadata.
    """

    model: "nn.Module"  # PyTorch model in eval mode
    device: str
    shape_spec: "ShapeSummary | None" = None
    feature_transforms: dict[str, "TransformChain"] | None = None
    target_transforms: dict[str, "TransformChain"] | None = None
    metadata: dict[str, str | int | float | bool | dict | list] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result from inference prediction.

    Simple dataclass holding prediction results.
    Following good practice of using dataclasses instead of bare returns.
    Access to model should be via predictor.model property, not in result.
    """

    predictions: "torch.Tensor | dict[str, torch.Tensor]"


