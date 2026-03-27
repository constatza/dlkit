"""DLKit shared domain kernel.

Shared result and state types consumed by both the runtime and interfaces layers.
Depends only on tools + stdlib/torch/lightning/tensordict.
"""

from .results import InferenceResult, OptimizationResult, TrainingResult
from .state import ModelState

__all__ = ["TrainingResult", "InferenceResult", "OptimizationResult", "ModelState"]
