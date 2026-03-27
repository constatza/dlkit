# Public API re-export shim — types moved to dlkit.domain
from dlkit.domain import InferenceResult, ModelState, OptimizationResult, TrainingResult

__all__ = ["TrainingResult", "InferenceResult", "OptimizationResult", "ModelState"]
