"""Public inference adapter over the runtime predictor subsystem."""

from dlkit.engine.inference import (
    CheckpointPredictor,
    IPredictor,
    PredictionOutput,
    PredictorConfig,
    PredictorError,
    PredictorNotLoadedError,
    evaluate_checkpoint,
    get_checkpoint_info,
    load_model,
    load_model_from_settings,
    log_evaluation_result,
    validate_checkpoint,
)

from .evaluate import evaluate
from .evaluate_multirun import ChildEvaluation, evaluate_multirun

__all__ = [
    # Main API
    "load_model",
    "validate_checkpoint",
    "get_checkpoint_info",
    # Predictor classes
    "CheckpointPredictor",
    "IPredictor",
    "PredictionOutput",
    "PredictorConfig",
    "load_model_from_settings",
    # Eval-only API
    "evaluate",
    "evaluate_checkpoint",
    "evaluate_multirun",
    "ChildEvaluation",
    "log_evaluation_result",
    # Exceptions
    "PredictorError",
    "PredictorNotLoadedError",
]
