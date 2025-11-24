"""Inference system for DLKit - NEW STATEFUL PREDICTOR ARCHITECTURE.

This module provides industry-standard inference with stateful predictors
that can be loaded once and reused for multiple predictions.

BREAKING CHANGES:
- Removed: infer() function (loaded model on every call)
- Removed: InferenceService class
- Added: load_predictor() - new primary API

Key Components:
- CheckpointPredictor: Stateful predictor object (load once, predict many)
- PredictorFactory: Factory for creating predictors
- ModelLoadingUseCase: Load model from checkpoint (expensive, once)
- InferenceExecutionUseCase: Execute inference (fast, many times)

Example:
    >>> from dlkit import load_predictor
    >>>
    >>> # Load once
    >>> predictor = load_predictor("model.ckpt", device="cuda")
    >>>
    >>> # Predict many times (no reloading!)
    >>> result1 = predictor.predict(input1)
    >>> result2 = predictor.predict(input2)
    >>> result3 = predictor.predict(input3)
    >>>
    >>> # Clean up
    >>> predictor.unload()

    >>> # Or with context manager
    >>> with load_predictor("model.ckpt") as predictor:
    ...     result = predictor.predict(inputs)
"""

from .api import load_predictor, validate_checkpoint, get_checkpoint_info
from .predictor import CheckpointPredictor, IPredictor, PredictorConfig
from .factory import PredictorFactory

__all__ = [
    # Main API
    "load_predictor",
    "validate_checkpoint",
    "get_checkpoint_info",
    # Predictor classes
    "CheckpointPredictor",
    "IPredictor",
    "PredictorConfig",
    "PredictorFactory",
]
