"""Configuration dataclasses for inference subsystem.

Simplified dataclasses without state machines or complex hierarchies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from dlkit.infrastructure.precision.strategy import PrecisionStrategy

if TYPE_CHECKING:
    import numpy as np
    from torch import nn

    from dlkit.common.shapes import ShapeSummary
    from dlkit.domain.transforms.chain import TransformChain


@dataclass(frozen=True, slots=True, kw_only=True)
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
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionOutput:
    """Typed prediction result for direct checkpoint inference."""

    predictions: torch.Tensor
    latents: tuple[torch.Tensor, ...] = ()
    raw: TensorDict | None = None

    def numpy(self) -> np.ndarray:
        """Return primary predictions as a NumPy array on CPU."""
        return self.predictions.detach().cpu().numpy()

    def __iter__(self):
        """Support unpacking as ``(predictions, *latents)``."""
        yield self.predictions
        yield from self.latents


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelState:
    """Model state container.

    Simplified from the earlier design - no state machine, just data.
    Holds loaded model and associated metadata.

    Attributes:
        model: PyTorch model in eval mode.
        device: Device string (e.g., "cpu", "cuda").
        shape_spec: Optional shape summary from dataset inference.
        feature_transforms: Named transform chains keyed by entry name.
        target_transforms: Named inverse transform chains keyed by entry name.
        metadata: Raw dlkit_metadata dict from checkpoint.
        feature_names: Ordered model-input entry names. Positional arg ``i``
            uses the transform registered under ``feature_names[i]``.
            Kwarg ``k`` uses the transform registered under ``k``.
        predict_target_key: Entry name whose chain is applied as inverse
            transform to the first model output at predict time.
    """

    model: nn.Module  # PyTorch model in eval mode
    device: str
    shape_spec: ShapeSummary | None = None
    feature_transforms: dict[str, TransformChain] | None = None
    target_transforms: dict[str, TransformChain] | None = None
    metadata: dict[str, str | int | float | bool | dict | list] = field(default_factory=dict)
    feature_names: tuple[str, ...] = field(default_factory=tuple)
    predict_target_key: str = ""
