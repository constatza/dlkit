"""Inference system for DLKit - Simplified Architecture.

Stateful predictors without hexagonal architecture overhead.
Direct functional API following industry standards.

Example:
    >>> from dlkit import load_model
    >>>
    >>> # Load once
    >>> predictor = load_model("model.ckpt", device="cuda")
    >>>
    >>> # Predict many times (no reloading!)
    >>> result1 = predictor.predict(input1)
    >>> result2 = predictor.predict(input2)
    >>> result3 = predictor.predict(input3)
    >>>
    >>> # Clean up
    >>> predictor.unload()

    >>> # Or with context manager
    >>> with load_model("model.ckpt") as predictor:
    ...     result = predictor.predict(inputs)
"""

from .api import load_model, validate_checkpoint, get_checkpoint_info
from .predictor import CheckpointPredictor, IPredictor, PredictorError, PredictorNotLoadedError
from .config import PredictorConfig, InferenceResult

__all__ = [
    # Main API
    "load_model",
    "validate_checkpoint",
    "get_checkpoint_info",
    # Predictor classes
    "CheckpointPredictor",
    "IPredictor",
    "PredictorConfig",
    "InferenceResult",
    # Exceptions
    "PredictorError",
    "PredictorNotLoadedError",
]
