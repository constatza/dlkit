"""User-facing inference namespace.

Thin re-export of the predictor-facing API.
"""

from dlkit.interfaces.inference import (
    CheckpointPredictor,
    IPredictor,
    PredictionOutput,
    PredictorConfig,
    PredictorError,
    PredictorNotLoadedError,
    get_checkpoint_info,
    load_model,
    load_model_from_settings,
    validate_checkpoint,
)

__all__ = [
    "load_model",
    "load_model_from_settings",
    "validate_checkpoint",
    "get_checkpoint_info",
    "CheckpointPredictor",
    "IPredictor",
    "PredictionOutput",
    "PredictorConfig",
    "PredictorError",
    "PredictorNotLoadedError",
]
